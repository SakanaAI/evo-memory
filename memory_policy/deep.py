import os
import pdb
import copy
import math
import numpy as np 
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, List

from omegaconf import OmegaConf, DictConfig
import hydra

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
from .base_dynamic import (
    DynamicMemoryPolicy, DynamicParamMemoryPolicy, 
    RecencyParams, AttentionParams,
    )


from .base_deep_components import (
    TokenEmbedding, ScoringNetwork, SelectionNetwork, DeepMemoryPolicyComponent)

from .shared import SynchronizableBufferStorage

def stat_fn_list(list_of_lists, index=None, 
            stats_to_return=['mean', 'std', 'above_mean', 'max', 'min']):
    if index is None:
        flat_list = np.array(
            [v for vs in list_of_lists for v in vs])
    else:
        flat_list = np.array(list_of_lists[index])
    to_return = {st: None for st in stats_to_return}
    if 'mean' in stats_to_return:
        to_return['mean'] = mean = np.mean(flat_list)
    if 'std' in stats_to_return:
        to_return['std'] = std = np.std(flat_list)
    if 'above_mean' in stats_to_return:
        to_return['above_mean'] = above_mean = np.mean(flat_list > mean)
    if 'max' in stats_to_return:
        to_return['max'] = max_v = np.max(flat_list)
    if 'min' in stats_to_return:
        to_return['min'] = np.min(flat_list)
    return to_return

def stat_fn_tensor(input_tensor, index=None, 
            stats_to_return=['mean', 'std', 'above_mean', 'max', 'min']):
    to_return = {st: None for st in stats_to_return}
    input_tensor = input_tensor.float()
    if 'mean' in stats_to_return:
        to_return['mean'] = mean = torch.mean(input_tensor).item()
    if 'std' in stats_to_return:
        to_return['std'] = std = torch.std(input_tensor).item()
    if 'above_mean' in stats_to_return:
        to_return['above_mean'] = above_mean = torch.mean(
            (input_tensor > mean).float()).item()
    if 'max' in stats_to_return:
        to_return['max'] = max_v = torch.max(input_tensor).item()
    if 'min' in stats_to_return:
        to_return['min'] = torch.min(input_tensor).item()
    return to_return

def reduce_stats(list_of_lists, stat_name,):
            
    stats = dict()
    mean_values = [np.mean(vs) for vs in list_of_lists]
    for i, m_v in enumerate(mean_values):
        stats_key_prefix = f'mem_stats/layer_id_{i}/{stat_name}'
        stats[stats_key_prefix] = m_v
    stats_key_prefix = f'mem_stats/overall/{stat_name}'
    stats[stats_key_prefix] = np.mean(mean_values)
    stats[stats_key_prefix + '_layers_std'] = np.std(mean_values)
    return stats




class DeepMP(DynamicParamMemoryPolicy):
    """Deep parameterized memory policy"""
    def __init__(
            self, 
            pop_size, 
            per_head, 
            per_layer,

            token_embedding: Union[TokenEmbedding, DictConfig],
            scoring_network: Union[ScoringNetwork, DictConfig],
            selection_criteria: Union[SelectionNetwork, DictConfig],
            lazy_param_num: bool = False,
            ):
        if isinstance(selection_criteria, DictConfig):
            selection_criteria: SelectionNetwork = hydra.utils.instantiate(
                selection_criteria)
        
        self._record_mask_based_sparsity = True
        cache_size = selection_criteria.get_cache_size()
        
        nn.Module.__init__(self=self)

        self.component_names = ['token_embedding', 'scoring_network', 
                                'selection_criteria']
        
        self.components_have_been_setup = False

        if isinstance(token_embedding, DictConfig):
            self.token_embedding: TokenEmbedding = hydra.utils.instantiate(
                token_embedding)
        else:
            self.token_embedding: TokenEmbedding = token_embedding
        
        if isinstance(scoring_network, DictConfig):
            self.scoring_network: ScoringNetwork = hydra.utils.instantiate(
                scoring_network)
        else:
            self.scoring_network: ScoringNetwork = scoring_network
        
        self.scoring_network.register_embedding(
            embedding_module=self.token_embedding)

        self.selection_criteria: SelectionNetwork = selection_criteria

        self.setup_properties()
        

        DynamicParamMemoryPolicy.__init__(
            self=self, 
            base_param_size=0, 
            pop_size=pop_size, 
            per_head=per_head, per_layer=per_layer, 
            additional_shared_params=0, 
            component_names=self.component_names,
            init_module=False,
            cache_size=cache_size,
            lazy_param_num=lazy_param_num,
            )
        if self.requires_position_ids:
            self._record_recency_stats = True
        self.scores_to_stats = {
            'token_embedding': ['mean', 'std', 'max', 'min'],
            'token_scores': ['mean', 'std', 'above_mean', 'max', 'min'],
            'retained_idxs': ['mean', 'max', 'min'],
        }
        self.stat_recorders = {
        }
        self.stat_getters = {
        }

        self.initialize_buffer_dicts_to_merge(
            buffers_to_merge=[], sub_buffer_storages=self.component_names)

    def setup_properties(self,):
        self._requires_attn_scores = self.true_for_any_component(
            'requires_attn_scores')
        self._requires_queries = self.true_for_any_component(
            'requires_queries')
        self._requires_position_ids = self.true_for_any_component(
            'requires_position_ids')
        self._is_diversity_based = self.true_for_any_component(
            'is_diversity_based')
    
    @property
    def record_mask_based_sparsity(self,):
        return self._record_mask_based_sparsity
    
    @record_mask_based_sparsity.setter
    def record_mask_based_sparsity(self, value):
        self._record_mask_based_sparsity = value
        
    def get_init_param_values(self,):
        return self.get_init_param_values_post_setup()
    
    
    
    
        
    
        
    def override_ema_coeff(self, new_ema):
        self.token_embedding.override_ema_coeff(new_ema=new_ema)
        self.scoring_network.override_ema_coeff(new_ema=new_ema)
        self.selection_criteria.override_ema_coeff(new_ema=new_ema)

    def update_layer_cache(
        self,
        layer_id,
        key_cache,
        value_cache,
        num_new_tokens,
        attn_weights=None,
        attn_mask=None,
        position_ids=None,
        threshold_shift: float = 0.0,
        ema_reduction_over: Optional[float] = None, 
        **kwargs,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        
        gathered_layer_params = self.get_layer_params(
            layer_id=layer_id)
        
        gathered_shared_params = self.get_additional_shared_params()
        
        token_embedding_params = self.get_params_for_token_embedding(
            params=gathered_layer_params,
            additional_params=gathered_shared_params,
            )
        
        scoring_network_params = self.get_params_for_scoring_network(
            params=gathered_layer_params,
            additional_params=gathered_shared_params,
            )
        
        selection_criteria_params = self.get_params_for_selection_criteria(
            params=gathered_layer_params,
            additional_params=gathered_shared_params,
            )

        key_cache, value_cache =  self.update_layer_cache_impl_(
            layer_id=layer_id,

            token_embedding_params=token_embedding_params,
            scoring_network_params=scoring_network_params,
            seletion_criteria_params=selection_criteria_params,

            key_cache=key_cache,
            value_cache=value_cache,
            num_new_tokens=num_new_tokens,
            attn_weights=attn_weights,
            attn_mask=attn_mask,
            position_ids=position_ids,
            threshold_shift=threshold_shift,
            ema_reduction_over=ema_reduction_over,
            analyze=False,
            **kwargs,
            )
        return key_cache, value_cache
    

    def analyze(
        self,
        layer_id,
        key_cache,
        value_cache,
        num_new_tokens,
        attn_weights=None,
        attn_mask=None,
        position_ids=None,
        threshold_shift: float = 0.0,
        ema_reduction_over: Optional[float] = None, 
        **kwargs,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        
        gathered_layer_params = self.get_layer_params(
            layer_id=layer_id)
        
        gathered_shared_params = self.get_additional_shared_params()
        
        token_embedding_params = self.get_params_for_token_embedding(
            params=gathered_layer_params,
            additional_params=gathered_shared_params,
            )
        
        scoring_network_params = self.get_params_for_scoring_network(
            params=gathered_layer_params,
            additional_params=gathered_shared_params,
            )
        
        selection_criteria_params = self.get_params_for_selection_criteria(
            params=gathered_layer_params,
            additional_params=gathered_shared_params,
            )

        
        key_cache, value_cache, analysis_dict = self.update_layer_cache_impl_(
            layer_id=layer_id,

            token_embedding_params=token_embedding_params,
            scoring_network_params=scoring_network_params,
            seletion_criteria_params=selection_criteria_params,

            key_cache=key_cache,
            value_cache=value_cache,
            num_new_tokens=num_new_tokens,
            attn_weights=attn_weights,
            attn_mask=attn_mask,
            position_ids=position_ids,
            threshold_shift=threshold_shift,
            ema_reduction_over=ema_reduction_over,
            analyze=True,
            **kwargs,
            )
        return key_cache, value_cache, analysis_dict
    
    def update_layer_cache_impl_(
        self,
        token_embedding_params,
        scoring_network_params,
        seletion_criteria_params,

        layer_id,
        key_cache,
        value_cache,
        num_new_tokens,
        attn_weights=None,
        attn_mask=None,
        position_ids=None,
        threshold_shift=0.0,
        analyze=False,
        
        analyze_get_full_jacobian=False,
        analyze_get_full_jacobian_for_layer: Optional[int] = None,
        analyze_get_full_jacobian_for_head: Optional[int] = None,
        
        **kwargs,
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        target_layer_id = 0
        bs, n_heads, num_all_tokens, n_embd = key_cache.shape
        device = key_cache.device

        if analyze:
            analysis_dict = {}

        new_sequences = num_all_tokens == num_new_tokens
        if num_all_tokens > num_new_tokens:
            if self.requires_position_ids:
                curr_pos_ids = self.cache_position_ids[layer_id]
            new_sequences = False
        else:
            new_sequences = True
        
        if attn_mask is not None:
            
            attn_mask = attn_mask.unsqueeze(-2)[..., -num_all_tokens:]
        if self.requires_position_ids:
            position_ids = self.process_position_ids(
                position_ids=position_ids, num_all_tokens=num_all_tokens, 
                num_new_tokens=num_new_tokens, attention_mask=attn_mask)
            if not new_sequences:
                curr_pos_ids = self.cache_position_ids[layer_id]
                position_ids = torch.concat(
                    [curr_pos_ids, position_ids], dim=-1)
                
        if self.requires_attn_scores:
            
            
            attn_weights = self.process_attn_weights(attn_weights=attn_weights)
            
            
            

        token_embedding = self.token_embedding.get_tokens_embedding(
            layer_id=layer_id,
            parameters=token_embedding_params,
            key_cache=key_cache,
            value_cache=value_cache,
            new_sequences=new_sequences,
            num_new_tokens=num_new_tokens,
            attn_weights=attn_weights,
            attn_mask=attn_mask,
            position_ids=position_ids,
            analyze=analyze,
            **kwargs,
            )
        if analyze:
            token_embedding.requires_grad = True
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        
        
        
        
        
        

        
        
        
        
        



        
        
        
        
        
        
        
            
        
            
            
            
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        token_scores = self.scoring_network.get_tokens_score(
            layer_id=layer_id,
            parameters=scoring_network_params,
            token_embeddings=token_embedding,
            new_sequences=new_sequences,
            num_new_tokens=num_new_tokens,
            attn_weights=attn_weights,
            attn_mask=attn_mask,
            position_ids=position_ids,
            analyze=analyze,
            **kwargs,
            )

        if analyze:
            
            token_scores_per_head = torch.unbind(token_scores[0], dim=0)
            
            embeddings_per_head = torch.unbind(token_embedding[0], dim=0)
            grad_analysis_dict = {}
            acc_grads_per_head = []
            latest_grads_per_head = []
            first_grads_per_head = []
            min_grads_per_head = []
            max_grads_per_head = []
            if analyze_get_full_jacobian:
                grads_per_head_per_out_token = []
            for i, (head_token_scores, head_token_embeddings) in enumerate(
                zip(token_scores_per_head, embeddings_per_head)):
                split_head_scores = torch.unbind(head_token_scores, dim=0)
                final_head = i == len(token_scores_per_head) - 1
                
                acc_scores_embedding_grads = torch.autograd.grad(
                        outputs=split_head_scores,
                        inputs=token_embedding,
                                                
                                                
                        retain_graph = True, 
                        )[0][0, i]
                acc_grads_per_head.append(acc_scores_embedding_grads)
                
                grads_per_out_token = []

                latest_token_scores = split_head_scores[-1]
                latest_scores_embedding_grads = torch.autograd.grad(
                        outputs=latest_token_scores,
                        inputs=token_embedding,
                                                
                                                
                        retain_graph = True, 
                        )[0][0, i]
                latest_grads_per_head.append(latest_scores_embedding_grads)
                first_token_scores = split_head_scores[0]
                first_scores_embedding_grads = torch.autograd.grad(
                        outputs=first_token_scores,
                        inputs=token_embedding,
                                                
                                                
                        retain_graph = True, 
                        )[0][0, i]
                first_grads_per_head.append(first_scores_embedding_grads)
                max_token_scores = head_token_scores.max()
                max_scores_embedding_grads = torch.autograd.grad(
                        outputs=max_token_scores,
                        inputs=token_embedding,
                                                
                                                
                        retain_graph = True, 
                        )[0][0, i]
                max_grads_per_head.append(max_scores_embedding_grads)
                min_token_scores = head_token_scores.min()
                min_scores_embedding_grads = torch.autograd.grad(
                        outputs=min_token_scores,
                        inputs=token_embedding,
                                                
                                                
                        retain_graph = True, 
                        )[0][0, i]
                min_grads_per_head.append(min_scores_embedding_grads)

                analyze_get_full_jacobian_head_check = (
                    analyze_get_full_jacobian_for_head == i) or (
                        analyze_get_full_jacobian_for_head is None)
                analyze_get_full_jacobian_layer_check = (
                    analyze_get_full_jacobian_for_layer == layer_id) or (
                        analyze_get_full_jacobian_for_layer is None)
                if (analyze_get_full_jacobian and 
                    analyze_get_full_jacobian_head_check and 
                    analyze_get_full_jacobian_layer_check):
                    print('Started computing full Jacobian for layer ' +
                          f'{layer_id}')
                    num_iters = len(split_head_scores)
                    for j, token_score in enumerate(split_head_scores):
                        print(f'{j}/{num_iters}')
                        scores_embedding_grads_per_token = torch.autograd.grad(
                            outputs=token_score,
                            inputs=token_embedding,
                            retain_graph = True,
                            )[0][0, i]
                        grads_per_out_token.append(
                            
                            scores_embedding_grads_per_token.cpu().numpy())
                    grads_per_head_per_out_token.append(
                        np.stack(grads_per_out_token, axis=0)) 
            grad_analysis_dict['scores_grads_per_head_acc'] = acc_grads_per_head
            grad_analysis_dict['scores_grads_per_head_latest'] = (
                latest_grads_per_head)
            grad_analysis_dict['scores_grads_per_head_first'] = (
                first_grads_per_head)
            grad_analysis_dict['scores_grads_per_head_max'] = (
                max_grads_per_head)
            grad_analysis_dict['scores_grads_per_head_min'] = (
                min_grads_per_head)
            if analyze_get_full_jacobian:
                grad_analysis_dict['all_scores_grads_per_head'] = (
                    grads_per_head_per_out_token)


        retained_idxs, new_mask = self.selection_criteria.select_new_tokens(
            layer_id=layer_id,
            parameters=seletion_criteria_params,
            token_scores=token_scores,
            new_sequences=new_sequences,
            num_new_tokens=num_new_tokens,
            attn_weights=attn_weights,
            attn_mask=attn_mask,
            position_ids=position_ids,
            threshold_shift=threshold_shift,
            analyze=analyze,
            **kwargs,
            )
        
        
        
        
        
        
        
            
        if not analyze:
            self.selection_criteria.filter_buffer_values(
                layer_id=layer_id,
                retained_idxs=retained_idxs,
                )
            
            self.scoring_network.filter_buffer_values(
                layer_id=layer_id,
                retained_idxs=retained_idxs,
                )

            self.token_embedding.filter_buffer_values(
                layer_id=layer_id,
                retained_idxs=retained_idxs,
                )

        if analyze:
            analysis_dict.update(dict(
                token_embedding=token_embedding,
                token_scores=token_scores,
                retained_idxs=retained_idxs,
                new_mask=new_mask,
                
                
                
                **grad_analysis_dict,
                
            ))
            analysis_dict.update(self.token_embedding.get_analysis_dict())
            analysis_dict.update(self.scoring_network.get_analysis_dict())
            analysis_dict.update(self.selection_criteria.get_analysis_dict())

                
        exp_retained_idxs = retained_idxs.unsqueeze(-1).expand(
            -1, -1, -1, n_embd)
        key_cache = torch.gather(key_cache, dim=-2, index=exp_retained_idxs)
        value_cache = torch.gather(value_cache, dim=-2, index=exp_retained_idxs)

        if self.requires_position_ids:
            if new_sequences:
                self.record_recency_stats(
                    layer_id=layer_id,
                    position_ids=self.cache_position_ids[layer_id],)
            if not analyze:
                self.cache_position_ids[layer_id] = torch.gather(
                    position_ids, dim=-1, index=retained_idxs)

        if self.auxiliary_loss_callback:
            
            
            
            self.auxiliary_loss.memory_policy_layer_callback(
                layer_id=layer_id,
                pop_idxs=self._flat_param_idxs,
                new_sequences=new_sequences,
                key_cache=key_cache,
                value_cache=value_cache, 
                dynamic_mask=new_mask,
                scoring_network_params=scoring_network_params
                )
            
        if self.record_eval_stats:
            self.record_dynamic_stats(
                layer_id=layer_id,
                cache_size=key_cache.shape[-2],
                new_sequences=new_sequences,
                )
            
            self.record_deep_stats(
                layer_id=layer_id,
                token_embedding=token_embedding,
                token_scores=token_scores,
                retained_idxs=retained_idxs,
                )
            
        if self.record_eval_stats or self.record_mask_based_sparsity:
            self.record_mask_dynamic_stats(
                layer_id=layer_id, cache_mask=new_mask)

        if analyze:
            return key_cache, value_cache, analysis_dict
        else:
            return key_cache, value_cache
    
    def true_for_any_component(self, property_name):
        true_for_token_embedding = getattr(self.token_embedding, property_name)
        true_for_scoring_network = getattr(self.scoring_network, property_name)
        true_for_selection_criteria = getattr(
            self.selection_criteria, property_name)
        return (true_for_token_embedding or
                true_for_scoring_network or 
                true_for_selection_criteria)
    
    @property
    def requires_attn_scores(self,):
        return self._requires_attn_scores
    
    @property
    def requires_queries(self,):
        return self._requires_queries
    
    @property
    def requires_position_ids(self,):
        return self._requires_position_ids
    
    @property
    def is_diversity_based(self,):
        return self._is_diversity_based
    
    
    def initialize_stat_objects_for(
        self,
        score_name,
        stats=['mean', 'std', 'above_mean', 'max', 'min'],
        ):
        for stat in stats:
            init_list = [[] for _ in range(self.num_memory_layers)]
            setattr(self,  f'{score_name}_{stat}', init_list)
        
        if score_name not in self.stat_recorders:
            def record_fn(element, layer_id):
                if isinstance(element, torch.Tensor):
                    rec_stats = stat_fn_tensor(
                        input_tensor=element, index=layer_id,
                        stats_to_return=stats)
                elif isinstance(element, List[list]):
                    rec_stats = stat_fn_list(
                        list_of_lists=element, index=layer_id,
                        stats_to_return=stats)
                else:
                    raise NotImplementedError
                for stat in stats:
                    getattr(self, f'{score_name}_{stat}')[layer_id].append(
                        rec_stats[stat]
                    )

            self.stat_recorders[score_name] = record_fn

        if score_name not in self.stat_getters:
            def get_fn(reset=True):
                stats_dict = dict()
                for stat in stats:
                    stat_name = f'{score_name}_{stat}'
                    list_of_lists = getattr(self, stat_name)
                    stats_dict.update(reduce_stats(
                        list_of_lists=list_of_lists, stat_name=stat_name))
                    if reset:
                        init_list = [[] for _ in range(self.num_memory_layers)]
                        setattr(self, stat_name, init_list)
                return stats_dict
            
            self.stat_getters[score_name] = get_fn
    
    def record_deep_stats(
            self,
            layer_id,
            **kwargs,
            
            
            
            ):
            for score_name, tensor in kwargs.items():
                self.stat_recorders[score_name](tensor, layer_id=layer_id)
    
    def get_deep_stats(self, reset=True):
            stats = dict()
            for score_name, getter_fn in self.stat_getters.items():
                stats.update(getter_fn(reset=reset))
            return stats

    def record_dynamic_stats(self, layer_id, cache_size, new_sequences):
        DynamicMemoryPolicy.record_dynamic_stats(
            self,
            layer_id=layer_id,
            cache_size=cache_size,
            new_sequences=new_sequences,
            )
    
    def record_mask_dynamic_stats(
            self, layer_id, 
            cache_mask, 
            ):
        
        unmasked_samples_per_head = cache_mask.to(torch.float32).sum(-1)
        
        self.dynamic_mask_sample_sparsity[layer_id].append( 
            torch.max(unmasked_samples_per_head, dim=-1)[0].mean().item())
        self.dynamic_mask_head_sparsity[layer_id].append(
            unmasked_samples_per_head.mean().item())

    def initialize_mask_based_sparsity(self, ):
        self.dynamic_mask_sample_sparsity = [[] for _ in range(
                self.num_memory_layers)]
        self.dynamic_mask_head_sparsity = [[] for _ in range(
                self.num_memory_layers)]                             

    def initialize_stat_objects(self, initialize_mask_spasity=True):
        DynamicMemoryPolicy.initialize_stat_objects(
            self,
            initialize_mask_spasity=initialize_mask_spasity,
            )
        for score_name, stats_to_record in self.scores_to_stats.items():
            self.initialize_stat_objects_for(
                score_name=score_name, stats=stats_to_record)
        
        
        
    def finalize_registration(self,):
        DynamicParamMemoryPolicy.finalize_registration(self,)
        self.register_model_to_components()

            
    
    def get_components_param_stats(self, reset=True) -> dict:
        if self.per_layer:
            raise NotImplementedError
        else:
            params = self.get_layer_params(layer_id=0)
            additional_params = self.get_additional_shared_params()
        stats = dict()
        for component_name in self.component_names:
            component_stats_prefix = f'mem_stats/{component_name}/'
            getter_name = 'get_params_for_' + component_name
            parameters = getattr(self, getter_name)(
                params=params, additional_params=additional_params)
            component: DeepMemoryPolicyComponent = getattr(self, component_name)
            component_stats = component.get_param_stats(
                parameters=parameters, reset=reset)
            for k, v in component_stats.items():
                stats[component_stats_prefix + k] = v
        return stats


    def get_param_stats(self, reset=True) -> dict:
        stats = self.get_components_param_stats(reset=reset)
        if self.record_eval_stats:
            stats.update(self.get_deep_stats(reset=False))
            
            stats.update(DynamicMemoryPolicy.get_param_stats(self, reset=reset))
        elif (self.record_eval_stats or self.record_mask_based_sparsity) and (
            reset):
            self.initialize_stat_objects()
        return stats

