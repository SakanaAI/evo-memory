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
from .base_dynamic import (
    DynamicMemoryPolicy, DynamicParamMemoryPolicy, 
    RecencyParams, AttentionParams, threshold_score_idxs,
    compute_recency_scores, compute_recency,
    )
from  .base_deep_components import (DeepMemoryPolicyComponent,
    ScoringNetwork, TokenEmbedding, SelectionNetwork, wrap_torch_initializer,
    ComponentOutputParams, true_for_all, true_for_any, get_matching_value, 
    call_for_all, get_property_list_from_elements)
from .deep_embedding_shared import Embedding, PositionalEmbedding

from stateless_parallel_modules import StatelessGeneralizedMLP


class BaseEmbeddingWrapper(TokenEmbedding, abc.ABC):
    def __init__(
            self,
            
            
            
            token_embedding: List[Union[TokenEmbedding, DictConfig]],
            embedding_dim: Optional[int] = None,
            wrapper_params: int = 0,
            wrapper_buffers: List[str] = [],
            output_params: Optional[ComponentOutputParams] = None,
            ):

        if isinstance(token_embedding, DictConfig):
            token_embedding: TokenEmbedding = hydra.utils.instantiate(
                token_embedding)
        else:
            assert isinstance(token_embedding, TokenEmbedding)
        
        te_requires_recomputation = token_embedding.requires_recomputation
        if output_params is None:
            output_params = ComponentOutputParams(
                requires_recomputation = te_requires_recomputation)
        else:
            if te_requires_recomputation:
                assert (output_params.requires_recomputation == True)

        per_layer = token_embedding.per_layer
        per_head = token_embedding.per_head
        shared = token_embedding.shared

        
        self._is_reduced_output = token_embedding.reduced_output
        self._param_scaling = token_embedding.get_param_scaling()

        self._num_wrapper_params = wrapper_params
        self._num_wrapped_embedding_params = token_embedding.get_param_size()
        self._num_net_params = (
            self._num_wrapper_params + self._num_wrapped_embedding_params)
        self._num_aux_params = 0
        self._wrapped_embedding_dim = token_embedding.get_embedding_dim()
        
        if embedding_dim is None:
            self.embedding_dim = self._wrapped_embedding_dim
        else:
            self.embedding = embedding_dim

        if not hasattr(self, '_requires_attn_scores'):
            self._requires_attn_scores = token_embedding.requires_attn_scores
        if not hasattr(self, '_requires_queries'):
            self._requires_queries = token_embedding.requires_queries
        if not hasattr(self, '_requires_position_ids'):
            self._requires_position_ids = token_embedding.requires_position_ids
        if not hasattr(self, '_is_diversity_based'):
            self._is_diversity_based = token_embedding.is_diversity_based
        

        TokenEmbedding.__init__(
            self,
            per_layer=per_layer,
            per_head=per_head,
            shared=shared,
            output_params=output_params,
            buffer_names=wrapper_buffers,
            initializer=0, 
            )
        
        self.wrapped_embedding = token_embedding
        self.register_sub_buffers_to_merge(
            sub_buffer_storages=[self.wrapped_embedding])
    
    def override_ema_coeff(self, new_ema):
        DeepMemoryPolicyComponent.override_ema_coeff(self=self, new_ema=new_ema)
        self.wrapped_embedding.override_ema_coeff(new_ema=new_ema)

    @abc.abstractmethod
    def get_tokens_embedding(
        self,
        layer_id,
        parameters,
        key_cache,
        value_cache,
        new_sequences,
        num_new_tokens,
        attn_weights=None,
        attn_mask=None,
        position_ids=None,
        **kwargs,
        ) -> torch.Tensor:
        raise NotImplementedError
        '''Builds a tensor representation for each KV cache token'''
        
        wrapped_emb_parameters, aux_params = self.split_net_and_aux_params(
            parameters=parameters,
            )
        
        expanded_dim = 1
    

    def get_embedding_dim(self,) -> int:
        return self._embedding_dim

    def net_param_size(self,) -> int:
        return self._num_net_params
    
    def aux_param_size(self,) -> int:
        return self._num_aux_params
    
    def get_param_initial(self,) -> Union[np.ndarray, torch.Tensor]:
        return self.wrapped_embedding.get_param_initial()
    
    def get_param_scaling(self,) -> Optional[Union[str, Tuple[float, float]]]:
        return self._param_scaling
    
    def register_new_memory_model(self, config, registration_kwargs):
        self.wrapped_embedding.register_new_memory_model(
            config=config,
            registration_kwargs=registration_kwargs,
            ) 
        TokenEmbedding.register_new_memory_model(
            self=self,
            config=config,
            registration_kwargs=registration_kwargs,
            )


    def finalize_registration(self,):
        self.wrapped_embedding.finalize_registration()
        
    def initialize_buffers(self,):
        self.wrapped_embedding.initialize_buffers() 
        
    def filter_buffer_values(
            self,
            layer_id: int,
            
            retained_idxs: torch.Tensor,
            ):
        self.wrapped_embedding.filter_buffer_values(
            layer_id=layer_id,
            retained_idxs=retained_idxs,
            )
    
    def reset_param_stats(self,):
        self.wrapped_embedding.reset_param_stats()

    def latest_stats(self,):
        self.wrapped_embedding.latest_stats()
    
    def get_param_stats(self, parameters, reset=True) -> dict:
        return self.wrapped_embedding.get_param_stats(
            parameters=parameters, reset=reset)
    
    def load_parameters(
            self,
            parameters: torch.Tensor,
            
            parallel_operations: Optional[int] = None,
            ):
        return self.wrapped_embedding.load_parameters(
            parameters=parameters, parallel_operations=parallel_operations)
    
    @property
    def reduced_output(self,):
        return self._is_reduced_output
        
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

class RecencyEmbeddingWrapper(BaseEmbeddingWrapper):
    def __init__(
            self,
            
            
            
            token_embedding: List[Union[TokenEmbedding, DictConfig]],
            recency_embedding: Embedding,
            wrapper_output_dim: Optional[int] = None,
            start_recency_from: int = 1,
            processing_layers: int = 0,
            joining_strategy: str = 'append',
            output_params: Optional[ComponentOutputParams] = None,
            ):
        self.start_recency_from = start_recency_from
        embed_dim = recency_embedding.embed_dim
        self.joining_strategy = joining_strategy.lower()
        assert self.joining_strategy in ['append', 'concat', 'add', 'mult']
        if self.joining_strategy == 'concat':
            self.joining_strategy = 'append'

        if isinstance(token_embedding, DictConfig):
            token_embedding: TokenEmbedding = hydra.utils.instantiate(
                token_embedding)
        else:
            assert isinstance(token_embedding, TokenEmbedding)

        if output_params is None:
            output_params = ComponentOutputParams(
                requires_recomputation = True)
        else:
            assert (output_params.requires_recomputation == True)

        self._wrapped_embedding_dim = token_embedding.get_embedding_dim()

        if wrapper_output_dim is None:
            if embed_dim is not None:
                self.wrapper_output_dim: int = embed_dim
            else:
                self.wrapper_output_dim: int = self._wrapped_embedding_dim
        else:
            self.wrapper_output_dim: int = wrapper_output_dim
        
        if embed_dim is None:
            self.embed_dim: int = self.wrapper_output_dim
            recency_embedding.set_embed_dim(embed_dim=self.embed_dim)
        else:
            self.embed_dim: int = embed_dim
        
        self._num_processing_layers = processing_layers
        if self._num_processing_layers == 0:
            assert self.embed_dim == self.wrapper_output_dim
        else:
            raise NotImplementedError

        if self.joining_strategy == 'append':
            self._embedding_dim = self._wrapped_embedding_dim + self.embed_dim
        elif self.joining_strategy == 'add' or self.joining_strategy == 'mult':
            assert self._wrapped_embedding_dim == self.embed_dim
            self._embedding_dim = self._wrapped_embedding_dim
            raise NotImplementedError
        else:
            raise NotImplementedError

        self._requires_position_ids = True

        BaseEmbeddingWrapper.__init__(
            self,
            token_embedding=token_embedding,
            embedding_dim=self.wrapper_output_dim,
            wrapper_params=0,
            wrapper_buffers=[],
            output_params=output_params,
            )
        
        self.recency_embedding = recency_embedding
    
    def get_tokens_embedding(
        self,
        layer_id,
        parameters,
        key_cache,
        value_cache,
        new_sequences,
        num_new_tokens,
        attn_weights=None,
        attn_mask=None,
        position_ids=None,
        **kwargs,
        ) -> torch.Tensor:
        '''Builds a tensor representation for each KV cache token'''
        
        wrapped_emb_parameters, aux_params = self.split_net_and_aux_params(
            parameters=parameters,
            )
        
        expanded_dim = 1
        
        output_embeddings = self.wrapped_embedding.get_tokens_embedding(
                layer_id=layer_id,
                parameters=wrapped_emb_parameters,
                key_cache=key_cache,
                value_cache=value_cache,
                new_sequences=new_sequences,
                num_new_tokens=num_new_tokens,
                attn_weights=attn_weights,
                attn_mask=attn_mask,
                position_ids=position_ids,
                **kwargs,
        )
        
        
        recencies = compute_recency(
            position_ids=position_ids,
            start_recency_from=self.start_recency_from,
            )
        
        
        
        
        recency_embeddings = self.recency_embedding(recencies)
        
        if not self._is_reduced_output:
            expanded_dim = output_embeddings.shape[-3]
            recency_embeddings = recency_embeddings.unsqueeze(
                        dim=-3).expand(-1, -1, expanded_dim, -1, -1)
        
        if self.joining_strategy == 'append':
            embeddings_to_join = [recency_embeddings, output_embeddings]
            embeddings = torch.concat(embeddings_to_join, dim=-1)
        elif self.joining_strategy == 'add':
            embeddings = recency_embeddings + output_embeddings
        elif self.joining_strategy == 'prod':
            embeddings = recency_embeddings*output_embeddings
        else:
            raise NotImplementedError
        
        embeddings = self.process_output(
                layer_id=layer_id,
                ema_coeff=self.ema_coeff,
                num_new_tokens=num_new_tokens,
                new_sequences=new_sequences,
                component_output=embeddings,
                aux_params=aux_params,
                attn_mask=attn_mask,
                **kwargs,
                )
        
        
        
        
        
        
        return embeddings

    def register_new_memory_model(self, config, registration_kwargs):
        self.wrapped_embedding.register_new_memory_model(
            config=config,
            registration_kwargs=registration_kwargs,
            ) 
        TokenEmbedding.register_new_memory_model(
            self=self,
            config=config,
            registration_kwargs=registration_kwargs,
            )


class AynmmetricFeaturesDeltaWrapper(BaseEmbeddingWrapper):
    def __init__(
            self,
            token_embedding: List[Union[TokenEmbedding, DictConfig]],
            output_params: Optional[ComponentOutputParams] = None,
            
            tokens_future: bool = True,
            ):
        
        self._wrapped_embedding_dim = token_embedding.get_embedding_dim()
        embedding_dim = self._wrapped_embedding_dim*2
        self.tokens_future = tokens_future
        BaseEmbeddingWrapper.__init__(
            self,
            token_embedding=token_embedding,
            embedding_dim=embedding_dim,
            wrapper_params=0,
            wrapper_buffers=[],
            output_params=output_params,
            )
        
        raise NotImplementedError
    
    def get_tokens_embedding(
        self,
        layer_id,
        parameters,
        key_cache,
        value_cache,
        new_sequences,
        num_new_tokens,
        attn_weights=None,
        attn_mask=None,
        position_ids=None,
        **kwargs,
        ) -> torch.Tensor:
        '''Builds a tensor representation for each KV cache token'''
        
        wrapped_emb_parameters, aux_params = self.split_net_and_aux_params(
            parameters=parameters,
            )

        
        output_embeddings = self.wrapped_embedding.get_tokens_embedding(
                layer_id=layer_id,
                parameters=wrapped_emb_parameters,
                key_cache=key_cache,
                value_cache=value_cache,
                new_sequences=new_sequences,
                num_new_tokens=num_new_tokens,
                attn_weights=attn_weights,
                attn_mask=attn_mask,
                position_ids=position_ids,
                **kwargs,
        )

        recencies = compute_recency(position_ids=position_ids)
        
        recency_embeddings = self.recency_embedding(recencies)
        if not self._is_reduced_output:
            expanded_dim = output_embeddings.shape[-3]
            recency_embeddings = recency_embeddings.unsqueeze(
                        dim=-3).expand(-1, -1, expanded_dim, -1, -1)
        
        if self.joining_strategy == 'append':
            embeddings_to_join = [recency_embeddings, output_embeddings]
            embeddings = torch.concat(embeddings_to_join, dim=-1)
        elif self.joining_strategy == 'add':
            embeddings = recency_embeddings + output_embeddings
        elif self.joining_strategy == 'prod':
            embeddings = recency_embeddings*output_embeddings
        else:
            raise NotImplementedError
            
        embeddings = self.process_output(
                layer_id=layer_id,
                ema_coeff=self.ema_coeff,
                num_new_tokens=num_new_tokens,
                new_sequences=new_sequences,
                component_output=embeddings,
                aux_params=aux_params,
                attn_mask=attn_mask,
                **kwargs,
                )
        
        
        return embeddings

    def register_new_memory_model(self, config, registration_kwargs):
        self.wrapped_embedding.register_new_memory_model(
            config=config,
            registration_kwargs=registration_kwargs,
            ) 
        TokenEmbedding.register_new_memory_model(
            self=self,
            config=config,
            registration_kwargs=registration_kwargs,
            )




class ParamOpWrapper(BaseEmbeddingWrapper):
    def __init__(
            self,
            
            
            
            token_embedding: List[Union[TokenEmbedding, DictConfig]],
            recency_embedding: Embedding,
            wrapper_output_dim: Optional[int] = None,
            processing_layers: int = 0,
            joining_strategy: str = 'append',
            output_params: Optional[ComponentOutputParams] = None,
            ):


        embed_dim = recency_embedding.embed_dim
        self.joining_strategy = joining_strategy.lower()
        assert self.joining_strategy in ['append', 'concat', 'add', 'mult']
        if self.joining_strategy == 'concat':
            self.joining_strategy = 'append'

        if isinstance(token_embedding, DictConfig):
            token_embedding: TokenEmbedding = hydra.utils.instantiate(
                token_embedding)
        else:
            assert isinstance(token_embedding, TokenEmbedding)

        if output_params is None:
            output_params = ComponentOutputParams(
                requires_recomputation = True)
        else:
            assert (output_params.requires_recomputation == True)

        self._wrapped_embedding_dim = token_embedding.get_embedding_dim()

        if wrapper_output_dim is None:
            if embed_dim is not None:
                self.wrapper_output_dim: int = embed_dim
            else:
                self.wrapper_output_dim: int = self._wrapped_embedding_dim
        else:
            self.wrapper_output_dim: int = wrapper_output_dim
        
        if embed_dim is None:
            self.embed_dim: int = self.wrapper_output_dim
            recency_embedding.set_embed_dim(embed_dim=self.embed_dim)
        else:
            self.embed_dim: int = embed_dim
        
        self._num_processing_layers = processing_layers
        if self._num_processing_layers == 0:
            assert self.embed_dim == self.wrapper_output_dim
        else:
            raise NotImplementedError

        if self.joining_strategy == 'append':
            self._embedding_dim = self._wrapped_embedding_dim + self.embed_dim
        elif self.joining_strategy == 'add' or self.joining_strategy == 'mult':
            assert self._wrapped_embedding_dim == self.embed_dim
            self._embedding_dim = self._wrapped_embedding_dim
            raise NotImplementedError
        else:
            raise NotImplementedError

        self._requires_position_ids = True

        BaseEmbeddingWrapper.__init__(
            self,
            token_embedding=token_embedding,
            embedding_dim=self.wrapper_output_dim,
            wrapper_params=0,
            wrapper_buffers=[],
            output_params=output_params,
            )
        
        self.recency_embedding = recency_embedding
    
    def get_tokens_embedding(
        self,
        layer_id,
        parameters,
        key_cache,
        value_cache,
        new_sequences,
        num_new_tokens,
        attn_weights=None,
        attn_mask=None,
        position_ids=None,
        **kwargs,
        ) -> torch.Tensor:
        '''Builds a tensor representation for each KV cache token'''
        
        wrapped_emb_parameters, aux_params = self.split_net_and_aux_params(
            parameters=parameters,
            )
        
        expanded_dim = 1
        
        output_embeddings = self.wrapped_embedding.get_tokens_embedding(
                layer_id=layer_id,
                parameters=wrapped_emb_parameters,
                key_cache=key_cache,
                value_cache=value_cache,
                new_sequences=new_sequences,
                num_new_tokens=num_new_tokens,
                attn_weights=attn_weights,
                attn_mask=attn_mask,
                position_ids=position_ids,
                **kwargs,
        )

        recencies = compute_recency(position_ids=position_ids)
        
        recency_embeddings = self.recency_embedding(recencies)
        if not self._is_reduced_output:
            expanded_dim = output_embeddings.shape[-3]
            recency_embeddings = recency_embeddings.unsqueeze(
                        dim=-3).expand(-1, -1, expanded_dim, -1, -1)
        
        if self.joining_strategy == 'append':
            embeddings_to_join = [recency_embeddings, output_embeddings]
            embeddings = torch.concat(embeddings_to_join, dim=-1)
        elif self.joining_strategy == 'add':
            embeddings = recency_embeddings + output_embeddings
        elif self.joining_strategy == 'prod':
            embeddings = recency_embeddings*output_embeddings
        else:
            raise NotImplementedError
            
        embeddings = self.process_output(
                layer_id=layer_id,
                ema_coeff=self.ema_coeff,
                num_new_tokens=num_new_tokens,
                new_sequences=new_sequences,
                component_output=embeddings,
                aux_params=aux_params,
                attn_mask=attn_mask,
                **kwargs,
                )
        
        
        return embeddings

    def register_new_memory_model(self, config, registration_kwargs):
        self.wrapped_embedding.register_new_memory_model(
            config=config,
            registration_kwargs=registration_kwargs,
            ) 
        TokenEmbedding.register_new_memory_model(
            self=self,
            config=config,
            registration_kwargs=registration_kwargs,
            )
