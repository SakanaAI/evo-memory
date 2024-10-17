import os
import pdb
import copy
import math
import numpy as np 
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, List

import abc
import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import LlamaPreTrainedModel
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from .base import MemoryPolicy, ParamMemoryPolicy


def compute_recency(position_ids, start_recency_from=1):
    most_recent_positions = position_ids[..., [-1]]
    
    cache_age = most_recent_positions - position_ids + start_recency_from
    return cache_age

def compute_recency_scores(position_ids, recency_exp, recency_coeff,):
    
    cache_age = compute_recency(position_ids=position_ids, start_recency_from=1)
    recency_scores = torch.pow(1/cache_age, exponent=recency_exp)
    return recency_scores*recency_coeff, recency_scores

def threshold_score_idxs(
        masked_full_scores,
        dynamic_thresh,
        
        
        preserve_order=True,
        cache_size=None,
        ):
    
    

    num_samples = masked_full_scores.shape[-1]
    if cache_size is not None and num_samples > cache_size:
        sorted_scores, indices = torch.topk(
            masked_full_scores, k=cache_size, sorted=True, dim=-1)
        sorted_scores = sorted_scores.flip(dims=(-1,))
        indices = indices.flip(dims=(-1,))
    else:
        sorted_scores, indices = torch.sort(
            masked_full_scores, descending=False, dim=-1)
    
    
    
    thresholded_scores = sorted_scores >= dynamic_thresh
    
    
    first_above_thresh = torch.sum(~thresholded_scores, dim=-1, 
                                    dtype=torch.long)

    discard_idx = torch.min(first_above_thresh)

    

    retained_idxs = indices[..., discard_idx:]

    
    new_mask = thresholded_scores[..., discard_idx:]

    if preserve_order:
        
        retained_idxs, _ = retained_idxs.sort(descending=False, dim=-1,)
    return retained_idxs, new_mask

@dataclass
class RecencyParams:
    recency_coeff: float 
    recency_exp: float 

@dataclass
class AttentionParams:
    attn_coeff: float 
    attn_ema_coeff: float 

    back_attn_coeff: float = 0 
    
    
    
    

class DynamicMemoryPolicy(MemoryPolicy):

    
    def __init__(self,
                 cache_size: Optional[int] = None,
                 init_module: bool = True,
                 ):
        
        MemoryPolicy.__init__(
            self,
            cache_size=cache_size,
            init_module=init_module,
            )
        
        self._record_mask_based_sparsity = False
        self._record_stats_per_head = False
        self._record_recency_stats = False
    
    @property
    def record_mask_based_sparsity(self,):
        return self._record_mask_based_sparsity
    
    @property
    def record_stats_per_head(self,):
        return self._record_mask_based_sparsity
    
    @record_mask_based_sparsity.setter
    def record_mask_based_sparsity(self, value):
        self._record_mask_based_sparsity = value

    @record_stats_per_head.setter
    def record_stats_per_head(self, value):
        self._record_stats_per_head = value

    def is_dynamic(self,):
        return True
    
    def select_max_score_idxs(
        self,
        masked_full_scores,
        cache_size,
        
        
        preserve_order=True,
        ):

        
        
        

        sorted_top_scores, retained_idxs = torch.topk(
            input=masked_full_scores, k=cache_size, largest=True,
            sorted=False,
        )

        if preserve_order:
            
            retained_idxs, _ = retained_idxs.sort(descending=False, dim=-1,)
        
        new_mask = torch.ones_like(retained_idxs)

        return retained_idxs, new_mask
    
    def threshold_score_idxs(
            self,
            masked_full_scores,
            dynamic_thresh,
            
            
            preserve_order=True,
            cache_size=None,
            ):
        
        retained_idxs, new_mask = threshold_score_idxs(
            masked_full_scores=masked_full_scores,
            dynamic_thresh=dynamic_thresh,
            preserve_order=preserve_order,
            cache_size=cache_size,
            )
        return retained_idxs, new_mask
    
    def select_new_dynamic_idxs(
            self,
            masked_full_scores,
            dynamic_thresh,
            cache_size,
            preserve_order=True,
            ):
        
        
        
        
        
        
        
        return self.threshold_score_idxs(
            masked_full_scores=masked_full_scores,
            dynamic_thresh=dynamic_thresh,
            preserve_order=preserve_order,
            cache_size=cache_size,
            )
    
    def need_new_dynamic_idxs(
            self,
            masked_full_scores,
            cache_size,
            ):
        if cache_size is None:
            return True
        else:
            num_samples = masked_full_scores.shape[-1]
            if num_samples > cache_size:
                return True
            else:
                return False


    
    def save_recency_params(self, recency_params: RecencyParams):
        self.recency_coeff = recency_params.recency_coeff
        self.recency_exp = recency_params.recency_exp
        self.use_recency_scores = False
        if self.recency_coeff is not None:
            if self.recency_coeff > 0:
                self.use_recency_scores = True

        print(f'Using recency score: {self.use_recency_scores}')


    def save_attention_params(self, attention_params: AttentionParams):

        self.attn_coeff = attention_params.attn_coeff
        self.attn_ema_coeff = attention_params.attn_ema_coeff

        self.use_full_attn_scores = False
        if self.attn_coeff is not None:
            if self.attn_coeff > 0:
                self.use_full_attn_scores = True
            
            
            

        self.back_attn_coeff = attention_params.back_attn_coeff
        

        self.use_back_attn_scores = False
        
        if self.back_attn_coeff is not None:
            if self.back_attn_coeff > 0:
                self.use_back_attn_scores = True
            
            
            
            
            
        self.use_attention_scores = (self.use_full_attn_scores or 
                                     self.use_back_attn_scores)
        if self.use_full_attn_scores and self.use_back_attn_scores:
            self.use_both_attn_scores = True
        else:
            self.use_both_attn_scores = False
            

        print(f'Using attention score: {self.use_attention_scores}')

        
    
    def process_position_ids(self, position_ids, num_all_tokens, num_new_tokens,
                             attention_mask):
        '''Replicates position ids for each head'''
        if position_ids is None:
            assert num_all_tokens == num_new_tokens
            assert attention_mask is not None
            
            position_ids = torch.cumsum(attention_mask, dim=-1) - 1
        position_ids = position_ids.unsqueeze(-2)
        return position_ids.expand(-1, self.num_heads, -1)
    
    def compute_recency_scores(self, position_ids, recency_exp, recency_coeff):
        return compute_recency_scores(
            position_ids=position_ids,
            recency_exp=recency_exp,
            recency_coeff=recency_coeff,
            )
    
    def initialize_cache_masks(self,):
        self.cache_masks = [None for _ in range(self.num_memory_layers)]

    def finalize_registration(self,):
        MemoryPolicy.finalize_registration(self,)
        self.initialize_cache_masks()
        self.initialize_stat_objects()

    
    def initialize_stat_objects(self, initialize_mask_spasity=True):
        self.dynamic_cache_sizes = [[] for _ in range(
                self.num_memory_layers)]
        self.final_dynamic_cache_sizes = [[] for _ in range(
                self.num_memory_layers)]
        
        if initialize_mask_spasity:
            self.initialize_mask_based_sparsity()
        if self._record_recency_stats:
            self.initialize_recency_stats()

    def initialize_mask_based_sparsity(self, ):
        self.dynamic_mask_sample_sparsity = [[] for _ in range(
                self.num_memory_layers)]
        self.dynamic_mask_head_sparsity = [[] for _ in range(
                self.num_memory_layers)]
        if self.record_stats_per_head:
            raise NotImplementedError
            self.dynamic_mask_head_sparsity_dicts = [{}]      

    def initialize_recency_stats(self,):
        self.recorded_final_recencies = [[] for _ in range(
                self.num_memory_layers)]
        self._record_recency_stats = True
        
    def record_recency_stats(self, layer_id, position_ids):
        if position_ids is not None:
            recency = compute_recency(position_ids=position_ids)
            self.recorded_final_recencies[layer_id].append(
                recency.float().mean().item())

    def get_param_stats(self, reset=True) -> dict:
        stats = dict()
        if self.record_eval_stats:
            if len(self.dynamic_cache_sizes[0]):
                all_final_cache_sizes = []
                for i in range(self.num_memory_layers):
                    stats_key_prefix = f'mem_stats/layer_id_{i}/'
                    stats[stats_key_prefix + 'dynamic_cache_sizes'] = np.mean(
                        self.dynamic_cache_sizes[i])
                    final_cache_sizes = (self.final_dynamic_cache_sizes[i] + 
                                         [self.dynamic_cache_sizes[i][-1]])
                    all_final_cache_sizes += final_cache_sizes
                    stats[stats_key_prefix + 'final_dynamic_cache_sizes'] = (
                        np.mean(final_cache_sizes))
                    
                stats_key_prefix = 'mem_stats/overall/'
                stats[stats_key_prefix + 'dynamic_cache_sizes'] = np.mean(
                    [v for vs in self.dynamic_cache_sizes for v in vs])
                stats[stats_key_prefix + 'final_dynamic_cache_sizes'] = np.mean(
                    [cs[-1] for cs in self.dynamic_cache_sizes])
                stats[stats_key_prefix + 'final_dynamic_cache_sizes'] = (
                    np.mean(all_final_cache_sizes))
        if self.record_eval_stats or self.record_mask_based_sparsity:
            if len(self.dynamic_mask_sample_sparsity[0]):
                for i in range(self.num_memory_layers):
                    stats_key_prefix = f'mem_stats/layer_id_{i}/'
                    stats[stats_key_prefix + 'unmasked_samples'] = np.mean(
                        self.dynamic_mask_sample_sparsity[i])
                    stats[stats_key_prefix + 'unmasked_samples_per_head'] = (
                        np.mean(self.dynamic_mask_head_sparsity[i]))
                    stats[stats_key_prefix + 'unmasked_sample_final'] = np.mean(
                        self.dynamic_mask_sample_sparsity[i][-1])
                    stats[stats_key_prefix +
                          'unmasked_sample_per_head_final'] = np.mean(
                              self.dynamic_mask_head_sparsity[i][-1])
                    
                    
                    
                stats_key_prefix = 'mem_stats/overall/'
                stats[stats_key_prefix + 'unmasked_samples'] = np.mean(
                    [v for vs in self.dynamic_mask_sample_sparsity for v in vs])
                stats[stats_key_prefix + 'unmasked_samples_per_head'] = np.mean(
                    [v for vs in self.dynamic_mask_head_sparsity for v in vs])
                stats[stats_key_prefix + 'unmasked_sample_final'] = np.mean(
                    [cs[-1] for cs in self.dynamic_mask_sample_sparsity])
                stats[stats_key_prefix + 'unmasked_sample_per_head_final'] = (
                    np.mean([cs[-1] for cs in self.dynamic_mask_head_sparsity]))
                
                
                
            if self._record_recency_stats:
                for i in range(self.num_memory_layers):
                    stats_key_prefix = f'mem_stats/layer_id_{i}/'
                    stats[stats_key_prefix + 'final_recencies'] = np.mean(
                        self.recorded_final_recencies[i])
                stats_key_prefix = 'mem_stats/overall/'
                stats[stats_key_prefix + 'final_recencies'] = np.mean(
                    [v for vs in self.recorded_final_recencies for v in vs])

            if reset:
                self.initialize_stat_objects()
        return stats
    
    def record_dynamic_stats(self, layer_id, cache_size, new_sequences=False):
        if new_sequences and len(self.dynamic_cache_sizes[layer_id]) > 0:
            self.final_dynamic_cache_sizes[layer_id].append(
                self.dynamic_cache_sizes[layer_id][-1])
        self.dynamic_cache_sizes[layer_id].append(int(cache_size))

    def record_mask_dynamic_stats(
            self, layer_id, 
            cache_mask, 
            ):
        
        unmasked_samples_per_head = cache_mask.to(torch.float32).sum(-1)
        
        self.dynamic_mask_sample_sparsity[layer_id].append( 
            torch.max(unmasked_samples_per_head, dim=-1)[0].mean().item())
        self.dynamic_mask_head_sparsity[layer_id].append(
            unmasked_samples_per_head.mean().item())

class DynamicParamMemoryPolicy(ParamMemoryPolicy, DynamicMemoryPolicy):

    def __init__(
            self, 
            base_param_size, 
            pop_size, 
            per_head, 
            per_layer,
            additional_shared_params=0,
            learnable_params: Optional[Dict[str, Union[str, tuple]]] = None,
            learned_params: Optional[Dict[str, bool]] = None,
            component_names: Optional[List[str]] = None,
            cache_size: Optional[int] = None,
            init_module: bool = True,
            lazy_param_num: bool = False,
            ):
        
        
        
        
        

        
        
        ParamMemoryPolicy.__init__(
            self, cache_size=cache_size, base_param_size=base_param_size, 
            pop_size=pop_size, per_head=per_head, per_layer=per_layer,
            additional_shared_params=additional_shared_params,
            learnable_params=learnable_params,
            learned_params=learned_params,
            component_names=component_names,
            init_module=init_module,
            lazy_param_num=lazy_param_num,
            )
        
    def is_dynamic(self,):
        return True
    
    def finalize_registration(self,):
        
        ParamMemoryPolicy.finalize_registration(self,)
        self.initialize_cache_masks()
        self.initialize_stat_objects()

    
