import os
import pdb
import copy
import math
import numbers
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, Callable, List
import hydra
from omegaconf import DictConfig

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import LlamaPreTrainedModel
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from .base import MemoryPolicy, ParamMemoryPolicy
from .base_dynamic import (
    DynamicMemoryPolicy, DynamicParamMemoryPolicy,
    RecencyParams, AttentionParams, threshold_score_idxs
)
from .base_deep_components import (
    ScoringNetwork, TokenEmbedding, SelectionNetwork, wrap_torch_initializer,
    ComponentOutputParams, reduce_ema_values)
from stateless_parallel_modules import StatelessGeneralizedMLP


def fft_ema_mask(window_length, ema_coeff, hop_length):
    '''Creates a mask mimicking an exponential moving average across the fft
       values. For consistent usage, ensure window_length == stride.'''
    discount_vector_exponents = torch.arange(
        start=window_length-1, end=-1, step=-1,)

    discount_vector = torch.pow(ema_coeff, discount_vector_exponents)
    rescale_factor = (1-ema_coeff)/(1-ema_coeff**hop_length)
    return discount_vector*rescale_factor


def fft_avg_mask(window_length):
    return torch.ones([window_length])/window_length


@dataclass
class STFTParams:
    n_fft: int
    hop_length: int
    window_fn: Optional[Union[
        Callable, np.ndarray, torch.Tensor, DictConfig]] = None
    pad_mode: str = 'constant'
    output_magnitudes: bool = True


class AttentionSpectrogram(TokenEmbedding):
    '''Representing each KV, via the unerlying freqs. for each token in the 
       attention matrix:
       Each token hits all its future tokens, given N new tokens we construct
       the spectrogram of all tokens based on the attention of the new tokens,
       we either store all future tokens spectrograms, prune old, or cumulate
       them with output_params (e.g., via EMA)'''

    def __init__(
            self,
            per_layer: bool,
            per_head: bool,
            shared: bool,
            output_params: ComponentOutputParams,
            stft_params: STFTParams,
            dtype: Optional[Union[str, torch.dtype]] = None,
    ):

        self.prev_attn_buffer: List[Optional[torch.Tensor]]
        assert output_params.requires_recomputation == True
        TokenEmbedding.__init__(
            self,
            per_layer=per_layer,
            per_head=per_head,
            shared=shared,
            output_params=output_params,
            buffer_names=['prev_attn_buffer'],
            dtype=dtype,



        )
        self.store_stft_params(stft_params=stft_params)

    def store_stft_params(self, stft_params: STFTParams):
        self.n_fft = stft_params.n_fft
        assert self.n_fft % 2 == 0

        self.window_length = self.n_fft
        self.stft_stride = stft_params.hop_length
        self.reduction_stride = self.stft_stride
        self.window_fn = stft_params.window_fn

        self.pad_attn_required = self.window_length - self.stft_stride
        self.base_n_fft = self.n_fft//2 + 1
        if self.window_fn is not None:
            if isinstance(self.window_fn, DictConfig):
                window = hydra.utils.call(self.window_fn, self.window_length)
            elif isinstance(self.window_fn, Callable):
                window = self.window_fn(self.window_length)
            else:
                window = self.window_fn
            assert window.shape[-1] == self.window_length
            self.register_buffer(
                'stft_window', tensor=window, persistent=False)
        else:
            self.stft_window = None
        self.output_magnitudes = stft_params.output_magnitudes
        self.stft_pad_mode = stft_params.pad_mode
        assert self.stft_pad_mode == 'constant', (
            'TODO: Deviating from constant padding might lead to inconsistent '
            'results, needs to be tested')

    def get_tokens_embedding(
        self,
        layer_id,
        parameters,
        key_cache,
        value_cache,
        new_sequences,
        num_new_tokens,
        attn_weights,
        attn_mask=None,
        position_ids=None,
        analyze=False,
        **kwargs,
    ) -> torch.Tensor:
        '''Builds a tensor representation for each KV cache token'''
        parameters, aux_params = self.split_net_and_aux_params(
            parameters=parameters)

        device = key_cache.device

        batch_size, n_heads, num_all_tokens, emb_dim = key_cache.shape
        batch_size, n_heads, num_new_tokens, num_all_tokens = attn_weights.shape

        if attn_mask is not None:

            attn_mask = attn_mask[..., -num_all_tokens:].unsqueeze(-2) > 0
        else:
            attn_mask = torch.ones([batch_size, 1, num_all_tokens],
                                   device=device, dtype=torch.bool)

        num_new_embeddings = num_new_tokens // self.stft_stride
        stride_carry_over = num_new_tokens % self.stft_stride
        if stride_carry_over > 0:

            raise NotImplementedError

        attn_weights = attn_weights.transpose(dim0=-2, dim1=-1)
        if self.pad_attn_required > 0:
            if new_sequences:

                pad_tuple = [self.pad_attn_required, 0]
                rel_attn_weights = F.pad(
                    input=attn_weights,
                    pad=pad_tuple,
                    mode=self.stft_pad_mode,
                )
            else:

                prev_attn_weights = self.prev_attn_buffer[layer_id]
                rel_prev_attn_weights = prev_attn_weights[
                    ..., -self.pad_attn_required:]
                pad_tuple = [0, 0, 0, num_new_tokens]

                padded_rel_prev_attn_weights = F.pad(
                    input=rel_prev_attn_weights,
                    pad=pad_tuple,
                    mode='constant',
                )

                rel_attn_weights = torch.concat(
                    [padded_rel_prev_attn_weights, attn_weights],
                    dim=-1,
                )

            if not analyze:
                self.prev_attn_buffer[layer_id] = attn_weights
        else:
            rel_attn_weights = attn_weights

        flat_rel_attn_weights = rel_attn_weights.flatten(
            start_dim=0, end_dim=-2)

        flat_attn_stft = torch.stft(
            input=flat_rel_attn_weights,
            n_fft=self.n_fft,
            hop_length=self.stft_stride,
            center=False,
            pad_mode='constant',
            normalized=False,
            onesided=True,
            return_complex=True,
            window=self.stft_window,
        )

        attn_stft = flat_attn_stft.view(
            batch_size, n_heads, num_all_tokens,
            self.base_n_fft, num_new_embeddings)

        attn_stft = attn_stft.permute(dims=[0, 1, 4, 2, 3])

        if self.output_magnitudes:
            attn_stft = attn_stft.abs()
        else:

            attn_stft = torch.view_as_real(attn_stft).flatten(
                start_dim=-2, end_dim=-1)

        if self._custom_dtype is not None:
            attn_stft = attn_stft.to(dtype=self.ptdtype)

        if self.output_past_non_reduced_history:

            past_attn_spectr = self.past_outputs_buffer[layer_id]

            pad_tuple = [0, 0, 0, num_new_tokens]
            padded_rel_prev_attn_weights = F.pad(
                input=rel_prev_attn_weights,
                pad=pad_tuple,
                mode='constant',
            )
            attn_stft = torch.concat([past_attn_spectr, attn_stft], dim=-3)
            if self.limit_past_history_size:
                attn_stft = attn_stft[
                    ..., -self.max_non_reduced_history_len:, :, :]
            embeddings = attn_stft

            if not analyze:
                self.past_outputs_buffer[layer_id] = attn_stft

        else:
            embeddings = attn_stft

        embeddings = self.process_output(
            layer_id=layer_id,
            ema_coeff=self.ema_coeff,
            num_new_tokens=num_new_tokens,
            new_sequences=new_sequences,
            component_output=embeddings,
            aux_params=aux_params,
            attn_mask=attn_mask,
            analyze=analyze,
            **kwargs,
        )

        return embeddings

    def get_embedding_dim(self,) -> int:

        if self.output_magnitudes:
            return self.base_n_fft
        else:
            return self.base_n_fft*2

    def net_param_size(self,) -> int:
        return 0

    @property
    def requires_attn_scores(self,):

        return True

    @property
    def requires_recomputation(self,):

        return True

    def filter_buffer_values(
            self,
            layer_id: int,

            retained_idxs: torch.Tensor):
        if self.pad_attn_required > 0:
            prev_attn_buffer: torch.Tensor = self.prev_attn_buffer[layer_id]
            prev_new_tokens = prev_attn_buffer.shape[-1]
            expanded_idxs = retained_idxs.unsqueeze(-1).expand(
                -1, -1, -1, prev_new_tokens)
            prev_attn_buffer = torch.gather(
                input=prev_attn_buffer,
                dim=-2,
                index=expanded_idxs,
            )
            self.prev_attn_buffer[layer_id] = prev_attn_buffer
        TokenEmbedding.filter_buffer_values(
            self=self,
            layer_id=layer_id,
            retained_idxs=retained_idxs)
