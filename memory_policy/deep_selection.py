import os
import pdb
import copy
import math
import numpy as np 
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict


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
from  .base_deep_components import SelectionNetwork, ComponentOutputParams


class DynamicSelection(SelectionNetwork):
    '''Replicates the default selection criteria of dynamic memory policies'''
    def __init__(self,
                 per_layer: bool,
                 per_head: bool,
                 shared: bool,
                 cache_size: Optional[int],
                 dynamic_thresh: float, 
                 ):
        SelectionNetwork.__init__(
            self,
            per_layer=per_layer,
            per_head=per_head,
            shared=shared,
            output_params=ComponentOutputParams(
                requires_recomputation=False,
                reduction_mode=None,
                ema_params=None,
                output_past_non_reduced_history=False,
                max_non_reduced_history_len=None,
            ),
            buffer_names=[],
            initializer=dynamic_thresh,
            )
        self.cache_size = cache_size

    def select_new_tokens(
        self,
        parameters,
        token_scores,
        new_sequences,
        num_new_tokens,
        attn_weights=None,
        attn_mask=None,
        position_ids=None,
        threshold_shift: float = 0.0,
        **kwargs,
        ) -> torch.Tensor:
        '''Produces indexes for the selected KV cache tokens and a selection
           mask.'''
        dynamic_thresh = parameters
        min_value = torch.finfo(token_scores.dtype).min
        max_value = torch.finfo(dynamic_thresh.dtype).max
        
        
        masked_full_scores = torch.where(
            attn_mask.bool(), token_scores, min_value)
        
        
        
        

        
        
        
        
        masked_full_scores[..., -1] = max_value
        retained_idxs, new_mask = threshold_score_idxs(
            masked_full_scores=masked_full_scores,
            dynamic_thresh=dynamic_thresh + threshold_shift,
            preserve_order=True,
            cache_size=self.cache_size,
            )
        
        
        return retained_idxs, new_mask
    
    def get_cache_size(self,) -> Optional[int]:
        return self.cache_size
    
    def net_param_size(self,) -> int:
        return 1
    
    def get_param_scaling(self,) -> Optional[Union[str, Tuple[float, float]]]:
        
        
        return 'exp'

class BinarySelection(SelectionNetwork):
    '''Mantains tokens when scores > 0 - can be probabilistic, based on a 
       logistic distribution instead'''
    
    def __init__(self,
                 per_layer: bool,
                 per_head: bool,
                 shared: bool,
                 cache_size: Optional[int],
                 is_probabilistic: bool = False,
                 temp: float = 1.0, 
                 learned_temp: bool = False, 
                 ):
        
        if learned_temp:
            assert is_probabilistic, (
                'If is not probabilistic, the temperature will not be used')
            self._needs_learned_temp = True
        else:
            self._needs_learned_temp = False
        self.is_probabilistic = is_probabilistic
        self.initial_temp = temp
        SelectionNetwork.__init__(
            self,
            per_layer=per_layer,
            per_head=per_head,
            shared=shared,
            output_params=ComponentOutputParams(
                requires_recomputation=False,
                reduction_mode=None,
                ema_params=None,
                output_past_non_reduced_history=False,
                max_non_reduced_history_len=None,
            ),
            buffer_names=[],
            initializer=temp,
            )
        self.cache_size = cache_size
        

    def select_new_tokens(
        self,
        parameters,
        token_scores,
        new_sequences,
        num_new_tokens,
        attn_weights=None,
        attn_mask=None,
        position_ids=None,
        threshold_shift: float = 0.0,
        **kwargs,
        ) -> torch.Tensor:
        '''Produces indexes for the selected KV cache tokens and a selection
           mask.'''
        
        
        
        
        
        min_value = torch.finfo(token_scores.dtype).min
        max_value = torch.finfo(token_scores.dtype).max

        if self.is_probabilistic:
            if self._needs_learned_temp:
                temp = parameters
            else:
                temp = self.initial_temp
            
            probabilities = F.sigmoid(masked_full_scores/temp)
            random_samples = torch.rand_like(probabilities)
            
            
            token_scores = (probabilities >= random_samples).to(
                probabilities.dtype) - 0.5

        masked_full_scores = torch.where(
            attn_mask.bool(), token_scores, min_value)
        masked_full_scores[..., -1] = max_value
        retained_idxs, new_mask = threshold_score_idxs(
            masked_full_scores=masked_full_scores,
            dynamic_thresh=threshold_shift,
            preserve_order=True,
            cache_size=self.cache_size,
            )
        
        
        return retained_idxs, new_mask
    
    def get_cache_size(self,) -> Optional[int]:
        return self.cache_size
    
    def net_param_size(self,) -> int:
        if self._needs_learned_temp:
            return 1
        else:
            return 0
    
    def get_param_scaling(self,) -> Optional[Union[str, Tuple[float, float]]]:
        
        
        return 'exp'


class TopKSelection(SelectionNetwork):
    '''Simply collects the top K scores with no thesholding'''
    def __init__(self,
                
                
                
                 cache_size: Optional[int],
                
                 ):
        SelectionNetwork.__init__(
            self,
            per_layer=False,
            per_head=False,
            shared=True,
            output_params=ComponentOutputParams(
                requires_recomputation=False,
                reduction_mode=None,
                ema_params=None,
                output_past_non_reduced_history=False,
                max_non_reduced_history_len=None,
            ),
            buffer_names=[],
            )
        self.cache_size = cache_size

    def select_new_tokens(
        self,
        parameters,
        token_scores,
        new_sequences,
        num_new_tokens,
        attn_weights=None,
        attn_mask=None,
        position_ids=None,
        **kwargs,
        ) -> torch.Tensor:
        '''Produces indexes for the selected KV cache tokens and a selection
           mask.'''
        
        num_samples = token_scores.shape[-1]
        if self.cache_size is not None and num_samples > self.cache_size:
            min_value = torch.finfo(token_scores.dtype).min
            masked_full_scores = torch.where(
                attn_mask.bool(), token_scores, min_value)
            _, retained_idxs = torch.topk(
                masked_full_scores, k=self.cache_size, sorted=False, dim=-1)
            retained_idxs, _ = retained_idxs.sort(descending=False, dim=-1,)
        else:
            retained_idxs = torch.arange(
                num_samples, device=token_scores.device,).view(
                    1, 1, num_samples).expand_as(token_scores)
        if self.cache_size is not None:
            attn_mask = attn_mask[..., -self.cache_size:]
        new_mask = torch.ones_like(retained_idxs, dtype=torch.bool)*attn_mask
        return retained_idxs, new_mask
    
    def get_cache_size(self,) -> Optional[int]:
        return self.cache_size
    
    def net_param_size(self,) -> int:
        return 0
    
    
    
    
    
