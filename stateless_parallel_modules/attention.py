import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, Callable


import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb)

from .base import (
    StatelessGeneralizedOperation, GeneralizedLinear,
    StatelessGeneralizedModule)
from utils import get_nonlinearity


class RotaryEmbedding(nn.Module):
    '''Based on transformers.models.llama.modeling_llama.LlamaRotaryEmbedding'''

    def __init__(
            self,
            dim,
            max_position_embeddings=2048,
            base=10000,
            device=None,
            scaling_factor=1.0,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (
            torch.arange(0, self.dim, 2,
                          dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(
                self.inv_freq)
        t = t / self.scaling_factor
        freqs = torch.outer(t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "_cos_cached", emb.cos().to(torch.get_default_dtype()),
            persistent=False)
        self.register_buffer(
            "_sin_cached", emb.sin().to(torch.get_default_dtype()),
            persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(
            device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @
                position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@dataclass
class StatelessAttentionParams:
    input_dim: Optional[int]
    # defaults to hidden dim.
    hidden_dim: Optional[int]
    output_dim: Optional[int]
    num_heads: int
    bias: bool
    max_position_id: int
    # rope params
    use_rope: bool
    rope_theta: float
    # None, forward, or backward
    masking_strategy: Optional[str]


class StatelessAttention(StatelessGeneralizedModule):
    def __init__(self,
                 attention_params: StatelessAttentionParams,):
        self.save_configs(attention_params=attention_params)
        StatelessGeneralizedModule.__init__(
            self=self,
            input_features=self.input_dim,
            output_features=self.output_dim,
            init_module=True,)
        if (self.input_dim is not None and
            self.output_dim is not None and
                self.hidden_dim is not None):

            self.instantiate_and_setup_ops(
                input_features=self.input_dim,
                hidden_features=self.hidden_dim,
                output_features=self.output_dim,
                preceding_module=None,
                default_output_features_mult=1,
            )

    def instantiate_and_setup_ops(
            self,
            input_features: Optional[int] = None,
            hidden_features: Optional[int] = None,
            output_features: Optional[int] = None,
            preceding_module=None,
            default_output_features_mult: int = 1,
            **kwargs,
    ):

        if (self.input_dim is None or
            self.output_dim is None or
                self.hidden_dim is None):

            self.instantiate_model(
                input_features=input_features,
                output_features=output_features,
                preceding_module=preceding_module,
                default_output_features_mult=default_output_features_mult,
            )
            self.input_dim = self.input_features
            self.output_dim = self.output_features
            if self.hidden_dim is None and hidden_features is not None:
                self.hidden_dim = hidden_features
            elif self.hidden_dim is None:
                print('Warning: hidden features not specified setting to ' +
                      f'{self.output_features} (output features)')
                self.hidden_dim = self.output_features

        self.head_dim = self.hidden_dim // self.num_heads
        if (self.head_dim * self.num_heads) != self.hidden_dim:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.linear_op = GeneralizedLinear()

        operation_kwargs = dict(
            bias=self.bias,
        )
        qkv_kwargs = dict(
            in_features=self.input_dim,
            out_features=self.hidden_dim*3,
        )
        o_kwargs = dict(
            in_features=self.hidden_dim,
            out_features=self.output_dim,
        )

        self._init_rope()

        self.setup_operations(
            operations=[self.linear_op, self.linear_op],
            operation_kwargs=operation_kwargs,
            operation_kwargs_overrides_list=[qkv_kwargs, o_kwargs],
        )

    def save_configs(self, attention_params: StatelessAttentionParams,):
        self.config = attention_params
        self.input_dim = attention_params.input_dim
        self.hidden_dim = attention_params.hidden_dim
        self.output_dim = attention_params.output_dim
        self.num_heads = attention_params.num_heads

        self.multiple_heads = self.num_heads > 1

        self.bias = attention_params.bias
        self.max_position_id = attention_params.max_position_id
        self.use_rope = attention_params.use_rope
        self.rope_theta = attention_params.rope_theta
        self.masking_strategy = attention_params.masking_strategy
        self.apply_causal_mask = False
        self.backward_causal_mask = False
        if self.masking_strategy is not None:
            self.apply_causal_mask = True
            if self.masking_strategy == 'backward':
                self.backward_causal_mask = True
            else:
                assert self.masking_strategy == 'forward'

    def _init_rope(self):
        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=self.max_position_id,
            base=self.rope_theta,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        *args,
        n_parallel_dimensions: Optional[int] = None,
        attn_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        *batch_dims, num_tokens, input_dim = inputs.size()

        weight, bias = self.parameters_per_layer[0]
        qkv_states = self.linear_op(
            input=inputs,
            weight=weight,
            bias=bias,
            parallel_operations=self.parallel_operations,
            n_parallel_dimensions=n_parallel_dimensions,
        )

        # 3 dimensional - flatten all batch dims
        qkv_states = qkv_states.flatten(start_dim=0, end_dim=-3)
        # 2 dimensional - flatten all batch dims
        position_ids = position_ids.flatten(start_dim=0, end_dim=-2)

        if not self.multiple_heads:
            # add singleton dimension to use attention over faster kernel
            qkv_states = qkv_states.unsqueeze_(1)

        query_states, key_states, value_states = torch.chunk(
            qkv_states, chunks=3, dim=-1)

        if self.multiple_heads:
            total_batch_dim = qkv_states.shape[0]
            query_states = query_states.view(
                total_batch_dim, num_tokens,
                self.num_heads, self.head_dim).transpose(-2, -3)
            key_states = key_states.view(
                total_batch_dim, num_tokens,
                self.num_heads, self.head_dim).transpose(-2, -3)
            value_states = value_states.view(
                total_batch_dim, num_tokens,
                self.num_heads, self.head_dim).transpose(-2, -3)

        if self.use_rope:
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin)

        if self.apply_causal_mask or attn_mask is not None:
            min_dtype = torch.finfo(inputs.dtype).min
            if self.apply_causal_mask:
                causal_mask = torch.ones(
                    (num_tokens, num_tokens),
                    device=key_states.device,
                    dtype=torch.bool,
                )
                if self.backward_causal_mask:
                    # one for all lower diagonal tokens to be masked
                    causal_mask = torch.tril(causal_mask, diagonal=-1)
                else:
                    # one for all upper diagonal tokens to be masked
                    causal_mask = torch.triu(causal_mask, diagonal=1)
            if attn_mask is not None:
                # In case num tokens < size of the attention mask
                inv_attn_mask = torch.logical_not(attn_mask[..., -num_tokens:])

                if self.apply_causal_mask:
                    causal_mask = torch.logical_or(inv_attn_mask, causal_mask)
                else:
                    causal_mask = inv_attn_mask
                causal_mask = causal_mask*min_dtype
            elif self.apply_causal_mask:
                causal_mask = causal_mask*min_dtype
            else:
                causal_mask = None

        attn_output = F.scaled_dot_product_attention(
            query=query_states,
            key=key_states,
            value=value_states,
            attn_mask=causal_mask,
            is_causal=False,
            scale=None,
        )
        if self.multiple_heads:
            attn_output = attn_output.transpose(-2, -3).contiguous()

            attn_output = attn_output.reshape(
                *batch_dims, num_tokens, self.hidden_dim)
        else:
            attn_output = attn_output.view(
                *batch_dims, num_tokens, self.hidden_dim)

        weight, bias = self.parameters_per_layer[1]
        attn_output = self.linear_op(
            input=attn_output,
            weight=weight,
            bias=bias,
            parallel_operations=self.parallel_operations,
            n_parallel_dimensions=n_parallel_dimensions,
        )
        return attn_output


class MonoHeadStatelessAttention(StatelessAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    # NOTE: hydra specification should have this as partial, since input_dim
    # might be unwieldy to always manually specify

    def __init__(self,
                 attention_params: StatelessAttentionParams,):
        # TODO
        self.save_configs(attention_params=attention_params)
        StatelessAttention.__init__(
            self=self, attention_params=attention_params)
        assert self.num_heads == 1

    def forward(
        self,
        inputs: torch.Tensor,
        *args,
        n_parallel_dimensions: Optional[int] = None,
        attn_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        *batch_dims, num_tokens, input_dim = inputs.size()

        # else:
        weight, bias = self.parameters_per_layer[0]
        qkv_states = self.linear_op(
            input=inputs,
            weight=weight,
            bias=bias,
            parallel_operations=self.parallel_operations,
            # handle reshaping internally
            n_parallel_dimensions=n_parallel_dimensions,
        )

        # 3 dimensional - flatten all batch dims
        qkv_states = qkv_states.flatten(start_dim=0, end_dim=-3)
        # 2 dimensional - flatten all batch dims
        position_ids = position_ids.flatten(start_dim=0, end_dim=-2)

        # total_batch_dim x num_tokens x head_dim (single head)
        query_states, key_states, value_states = torch.chunk(
            qkv_states, chunks=3, dim=-1)

        if self.use_rope:
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin)

        if self.apply_causal_mask:
            causal_mask = torch.ones(
                (num_tokens, num_tokens),
                device=key_states.device,
                dtype=torch.bool,
            )
            if self.backward_causal_mask:
                # one for all lower diagonal tokens to be masked
                causal_mask = torch.tril(causal_mask, diagonal=-1)
            else:
                # one for all upper diagonal tokens to be masked
                causal_mask = torch.triu(causal_mask, diagonal=1)
        if attn_mask is not None:
            # In case num tokens < size of the attention mask
            inv_attn_mask = torch.logical_not(attn_mask[..., -num_tokens:])

            if self.apply_causal_mask:
                causal_mask = torch.logical_or(inv_attn_mask, causal_mask)
            else:
                causal_mask = inv_attn_mask

            min_dtype = torch.finfo(inputs.dtype).min
            causal_mask = causal_mask*min_dtype
        elif self.apply_causal_mask:
            causal_mask = causal_mask*min_dtype
        else:
            causal_mask = None

        attn_output = F.scaled_dot_product_attention(
            query=query_states,
            key=key_states,
            value=value_states,
            attn_mask=causal_mask,
            is_causal=False,
            scale=None,  # defaults to 1/root(head_dim)
        )

        attn_output = attn_output.view(
            *batch_dims, num_tokens, self.hidden_dim)

        weight, bias = self.parameters_per_layer[1]
        attn_output = self.linear_op(
            input=attn_output,
            weight=weight,
            bias=bias,
            parallel_operations=self.parallel_operations,
            n_parallel_dimensions=n_parallel_dimensions,
        )

        return attn_output
