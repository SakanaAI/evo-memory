import os
import pdb
import copy
import math
import numbers
import abc
import numpy as np 
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, Callable, List

from omegaconf import OmegaConf, DictConfig
import hydra

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import LlamaPreTrainedModel
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from .base import MemoryPolicy, ParamMemoryPolicy
from .base_dynamic import compute_recency

from  .base_deep_components import (
    ScoringNetwork, TokenEmbedding, SelectionNetwork, wrap_torch_initializer,
    ComponentOutputParams,)

from .deep_embedding_shared import Embedding, PositionalEmbedding

from stateless_parallel_modules import StatelessGeneralizedMLP

def convert_to_tensor(
        el: Union[List[float], np.ndarray, torch.Tensor],
        ) -> torch.Tensor:
    if isinstance(el, torch.Tensor):
        return el
    else:
        el = torch.tensor(el)
    return el


class RecencyExponents(TokenEmbedding):
    '''Representing each KV, via a polynomial vector of its recency'''
    def __init__(
            self,
            per_layer: bool, 
            per_head: bool, 
            shared: bool,
            initial_exponents: Union[List[float], np.ndarray, torch.Tensor],
            dtype: Optional[Union[str, torch.dtype]] = None
            ):
        initial_exponents = convert_to_tensor(initial_exponents)
        assert len(initial_exponents.shape) == 1
        self._num_recency_exponents = initial_exponents.shape[-1]
        TokenEmbedding.__init__(
            self,
            per_layer=per_layer,
            per_head=per_head,
            shared=shared,
            output_params=ComponentOutputParams(
                requires_recomputation=False,),
            buffer_names=[],
            initializer=initial_exponents,
            dtype=dtype,
            )
        self.initial_recency_exponents = initial_exponents

    def get_tokens_embedding(
        self,
        layer_id,
        parameters,
        key_cache,
        value_cache,
        new_sequences,
        num_new_tokens,
        position_ids,
        attn_mask=None,
        **kwargs,
        ) -> torch.Tensor:
        '''Builds a tensor representation for each KV cache token'''
        
        exponents = parameters

        
        unsqueezed_exponents = exponents.unsqueeze(dim=-2)
        

        cache_recencies = compute_recency(position_ids=position_ids)
        unsqueezed_recencies = cache_recencies.unsqueeze(dim=-1)
        
        embeddings = torch.pow(
            input=unsqueezed_recencies, exponent=unsqueezed_exponents)
        
        if self._custom_dtype is not None:
            embeddings = embeddings.to(dtype=self.ptdtype)

        
        embeddings = self.process_output(
            layer_id=layer_id,
            ema_coeff=self.ema_coeff,
            num_new_tokens=num_new_tokens,
            new_sequences=new_sequences,
            component_output=embeddings,
            attn_mask=attn_mask,
            **kwargs,
            )

        return embeddings
    
    def get_embedding_dim(self,) -> int:
        return self._num_recency_exponents
    
    def net_param_size(self,) -> int:
        return self._num_recency_exponents
    
    def aux_param_size(self) -> int:
        return 0
    
    def get_net_params_stats(self, parameters: torch.Tensor):
        stats = dict()
        learned_exps = parameters.split(split_size=1, dim=-1)
        for i, learned_exp in enumerate(learned_exps):
            stats[f'net_params/rec_exp_{i}'] = learned_exp.mean().item()
        return stats

    @property
    def requires_position_ids(self,):
        
        return True
    
    @property
    def requires_recomputation(self,):
        
        return False

    @property
    def reduced_output(self,):
        return True
    
    def net_param_size(self,) -> int:
        return self._num_recency_exponents


class NormalizedRecencyExponents(TokenEmbedding):
    '''Representing each KV, via a polynomial vector of its normalized 
       recency - a score ranging from 1 to 0 (most recent to oldest possible
       i.e., max_position_id)'''
    def __init__(
            self,
            per_layer: bool, 
            per_head: bool, 
            shared: bool,
            max_position_id: int,
            initial_exponents: Union[List[float], np.ndarray, torch.Tensor],
            only_positive_exponents: bool = True,
            dtype: Optional[Union[str, torch.dtype]] = None
            ):
        initial_exponents = convert_to_tensor(initial_exponents)
        assert len(initial_exponents.shape) == 1
        self._num_recency_exponents = initial_exponents.shape[-1]
        self.max_position_id = max_position_id
        self.only_positive_exponents = only_positive_exponents

        if only_positive_exponents:
            assert torch.all(initial_exponents > 0)
            log_initial_exponents = torch.log(
                initial_exponents)
            initializer = log_initial_exponents
        else:
            initializer = initial_exponents

        TokenEmbedding.__init__(
            self,
            per_layer=per_layer,
            per_head=per_head,
            shared=shared,
            output_params=ComponentOutputParams(
                requires_recomputation=False,),
            buffer_names=[],
            initializer=initializer,
            dtype=dtype,
            )

    def get_tokens_embedding(
        self,
        layer_id,
        parameters,
        key_cache,
        value_cache,
        new_sequences,
        num_new_tokens,
        position_ids,
        attn_mask=None,
        **kwargs,
        ) -> torch.Tensor:
        '''Builds a tensor representation for each KV cache token'''
        
        if self.only_positive_exponents:
            exponents = torch.exp(parameters)
        else:
            exponents = parameters

        
        unsqueezed_exponents = exponents.unsqueeze(dim=-2)
            
        
        
        cache_recencies = compute_recency(
            position_ids=position_ids)/self.max_position_id
        
        
        
        cache_recencies = 1 - cache_recencies
        
        unsqueezed_recencies = cache_recencies.unsqueeze(dim=-1)
        
        embeddings = torch.pow(
            input=unsqueezed_recencies, exponent=unsqueezed_exponents)
        
        if self._custom_dtype is not None:
            embeddings = embeddings.to(dtype=self.ptdtype)
            
        embeddings = self.process_output(
            layer_id=layer_id,
            ema_coeff=self.ema_coeff,
            num_new_tokens=num_new_tokens,
            new_sequences=new_sequences,
            component_output=embeddings,
            attn_mask=attn_mask,
            **kwargs,
            )
        return embeddings
    
    def get_embedding_dim(self,) -> int:
        return self._num_recency_exponents
    
    def net_param_size(self,) -> int:
        return self._num_recency_exponents
    
    def aux_param_size(self) -> int:
        return 0
    
    def get_net_params_stats(self, parameters: torch.Tensor):
        stats = dict()
        learned_exps = parameters.split(split_size=1, dim=-1)
        for i, learned_exp in enumerate(learned_exps):
            stats[f'net_params/rec_exp_{i}'] = learned_exp.mean().item()
        return stats

    @property
    def requires_position_ids(self,):
        
        return True
    
    @property
    def requires_recomputation(self,):
        
        return False

    @property
    def reduced_output(self,):
        return True
    
    def net_param_size(self,) -> int:
        return self._num_recency_exponents
    

