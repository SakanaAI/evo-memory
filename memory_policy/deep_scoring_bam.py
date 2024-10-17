import os
import pdb
import copy
import math
import numbers
import numpy as np 
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, Callable, List


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
from  .base_deep_components import (
    ScoringNetwork, TokenEmbedding, SelectionNetwork, wrap_torch_initializer,
    ComponentOutputParams)
from stateless_parallel_modules import StatelessGeneralizedMLP


def make_scaled_one_hot_init(
        idxs_to_scale: dict,
        idxs_to_ones: Union[List[int], np.ndarray, torch.Tensor]):
    def _init_fn(shape):
        tensor = torch.zeros(shape)
        for idx in idxs_to_ones:
            tensor[..., idx] = 1.0
        for idx, value in idxs_to_scale.items():
            tensor[..., int(idx)] = float(value)
            
        return tensor
    return _init_fn

class MLPScoring(ScoringNetwork):
    '''MLP scoring layer, producing score as the NN output combination of the
       embeddings.'''
    def __init__(
            self,
            per_layer: bool, 
            per_head: bool, 
            shared: bool,
            output_params: ComponentOutputParams,
            hidden_features: Optional[int],
            hidden_depth: int,
            bias: bool,
            non_linearity: Optional[Union[str, Callable]] = 'relu',
            initializer: numbers.Number = 0,
            residual: bool = True,
            residual_first: bool = False,
            
            ):
        ScoringNetwork.__init__(
            self,
            per_layer=per_layer,
            per_head=per_head,
            shared=shared,
            output_params=output_params,
            buffer_names=['past_scores'],
            initializer=initializer,
            )
        self.hidden_features = hidden_features
        self.hidden_depth = hidden_depth
        self.bias = bias
        self.non_linearity = non_linearity
        self.residual = residual
        self.residual_first = residual_first

    def register_embedding(self, embedding_module: TokenEmbedding):
        ScoringNetwork.register_embedding(
            self=self,
            embedding_module=embedding_module,
            )
        if self.hidden_features is None:
            self.hidden_features=self.input_embedding_dim
        self.mlp = StatelessGeneralizedMLP(
            input_features=self.input_embedding_dim,
            hidden_features=self.hidden_features,
            output_features=1,
            hidden_depth=self.hidden_depth,
            bias=self.bias,
            non_linearity=self.non_linearity,
            residual=self.residual,
            residual_first=self.residual_first,
        )
        self.mlp_base_parameters = self.mlp.total_base_parameter_dims
    
    def get_tokens_score(
        self,
        layer_id,
        parameters,
        token_embeddings: torch.Tensor,
        new_sequences,
        num_new_tokens,
        attn_weights=None,
        attn_mask=None,
        position_ids=None,
        
        **kwargs,
        ) -> torch.Tensor:
        '''Produces score for each KV cache token embedding'''

        if not self.requires_recomputation:
            token_embeddings = token_embeddings[..., -num_new_tokens:, :]

        if self.is_reduced_input:
            batch_size, n_heads, n_out_tokens, emb_dim = token_embeddings.shape
        else:
            batch_size, n_heads, non_reduced_outputs, n_out_tokens, emb_dim = (
                token_embeddings.shape)
            token_embeddings = token_embeddings.flatten(start_dim=2, end_dim=3)
            
        
        
        n_out_tokens = n_out_tokens
        parallel_operations = batch_size
        

        if self.shared and self.per_head:
            
            parallel_operations = parallel_operations*n_heads
            token_embeddings = token_embeddings.flatten(start_dim=0, end_dim=1)
        else:
            
            token_embeddings = token_embeddings.flatten(start_dim=1, end_dim=2)
        
        
        
        self.mlp.load_parameters(
            parameters=parameters,
            parallel_operations=parallel_operations,
            )
        scores = self.mlp(inputs=token_embeddings)
        
        if self.is_reduced_input:
            scores = scores.view(batch_size, n_heads, n_out_tokens)
        else:
            scores = scores.view(
                batch_size, n_heads, non_reduced_outputs, n_out_tokens)

        if not self.requires_recomputation:
            if not new_sequences:
                
                past_scores: torch.Tensor = self.past_scores[layer_id]
                scores = torch.concat([past_scores, scores], dim=-1)
            self.past_scores[layer_id] = scores
            
        if (self.reduction_mode is not None) and (not self.is_reduced_input):
            
            scores = self.process_output(
                layer_id=layer_id,
                ema_coeff=self.ema_coeff,
                num_new_tokens=num_new_tokens,
                new_sequences=new_sequences,
                component_output=scores,
                **kwargs,
                )
        return scores
    
    def filter_buffer_values(
            self,
            layer_id: int,
            
            retained_idxs: torch.Tensor,
            ):
        ScoringNetwork.filter_buffer_values(
            self=self,
            layer_id=layer_id,
            retained_idxs=retained_idxs,
            )
        if not self.requires_recomputation:
            

            past_scores: torch.Tensor = self.past_scores[layer_id]

            
            
            

            
            
            

            self.past_scores[layer_id] = torch.gather(
                input=past_scores, dim=-1, index=retained_idxs)
    
    def net_param_size(self,) -> int:
        return self.mlp_base_parameters
    

class LinearScoring(MLPScoring):
    '''Linear scoring layer, producing score as a linear combination of the
       embeddings.'''
    def __init__(
            self,
            per_layer: bool, 
            per_head: bool, 
            shared: bool,
            output_params: ComponentOutputParams,
            bias: bool,
            
            initializer: numbers.Number = 0,
            ):
        MLPScoring.__init__(
            self,
            per_layer=per_layer,
            per_head=per_head,
            shared=shared,
            output_params=output_params,
            hidden_features=0,
            hidden_depth=1,
            bias=bias,
            non_linearity=None,
            initializer=initializer,
            )
        self.bias = bias

