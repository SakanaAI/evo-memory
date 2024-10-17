import os
import pdb
import copy
import math
import numbers
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
    RecencyParams, AttentionParams, threshold_score_idxs
    )
from  .base_deep_components import (
    ScoringNetwork, TokenEmbedding, SelectionNetwork, wrap_torch_initializer,
    ComponentOutputParams)
from stateless_parallel_modules import (
    StatelessGeneralizedMLP, StatelessGeneralizedModule,
    StatelessGeneralizedOperation)
from utils import get_nonlinearity


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
            dtype: Optional[Union[str, torch.dtype]] = None,
            
            ):
        
        ScoringNetwork.__init__(
            self,
            per_layer=per_layer,
            per_head=per_head,
            shared=shared,
            output_params=output_params,
            buffer_names=['past_scores'],
            initializer=initializer,
            dtype=dtype,
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
        parameters: torch.Tensor,
        token_embeddings: torch.Tensor,
        new_sequences,
        num_new_tokens,
        attn_weights=None,
        attn_mask=None,
        position_ids=None,
        
        **kwargs,
        ) -> torch.Tensor:
        '''Produces score for each KV cache token embedding'''
        
        if self._custom_dtype is not None:
            parameters = parameters.to(dtype=self.ptdtype)
            token_embeddings = token_embeddings.to(dtype=self.ptdtype)

        if not self.requires_recomputation:
            token_embeddings = token_embeddings[..., -num_new_tokens:, :]

        
        
        
        
        
            
            
        
        
        
        batch_size, n_heads = token_embeddings.shape[:2]
        parallel_operations = batch_size
        

        
        
        
        
        
        
        
        
        if self.shared and self.per_head:
            parallel_operations = parallel_operations*n_heads
            
            
            n_parallel_dimensions = 2
            
        else:
            
            n_parallel_dimensions = 1
            

        
        

        self.mlp.load_parameters(
            parameters=parameters,
            parallel_operations=parallel_operations,
            )
        scores: torch.Tensor = self.mlp(
            inputs=token_embeddings,
            n_parallel_dimensions=n_parallel_dimensions,
            
            )
        
        
        
        scores = scores.squeeze_(dim=-1)
        
        
        
        
        
        
        

        if not self.requires_recomputation:
            if not new_sequences:
                
                past_scores: torch.Tensor = self.past_scores[layer_id]
                scores = torch.concat([past_scores, scores], dim=-1)
            self.past_scores[layer_id] = scores
            
        
        
        scores = self.process_output(
            layer_id=layer_id,
            ema_coeff=self.ema_coeff,
            num_new_tokens=num_new_tokens,
            new_sequences=new_sequences,
            component_output=scores,
            attn_mask=attn_mask,
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


class GeneralizedScoring(ScoringNetwork):
    '''Producing score as the NN output combination of the embeddings using
       arbitrary stateless generalized modules.'''
    def __init__(
            self,
            per_layer: bool, 
            per_head: bool, 
            shared: bool,
            output_params: ComponentOutputParams,
            
            stateless_modules_list: List[
                Union[DictConfig, 
                      StatelessGeneralizedModule]],
            
            initializer: numbers.Number = 0,
            
            
            residual: Union[bool, List[bool]] = True,

            
            
            
            mult: Union[bool, List[bool]] = False,
            mult_nonlinearity: Optional[Union[str, Callable]] = None,
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
        
        self.stateless_modules_list: List[StatelessGeneralizedModule] = []

        for stateless_module in stateless_modules_list:
            if isinstance(stateless_module, DictConfig):
                stateless_module: StatelessGeneralizedModule = (
                    hydra.utils.instantiate(stateless_module))
            else:
                stateless_module: StatelessGeneralizedModule = stateless_module
            self.stateless_modules_list.append(stateless_module)
        
        self._num_stateless_modules = len(self.stateless_modules_list)
        assert self._num_stateless_modules > 0

        if isinstance(residual, bool):
            self.residual_list: List[bool] = [residual for _ in range(
                self._num_stateless_modules)]
        else:
            self.residual_list: List[bool] = residual
            if self.residual_list[-1] == True:
                print('Warning: residual set to true in the final layer, ' +
                      'overriding it with False (out_dims must be 1)')
        self.residual_list[-1] = False

        if isinstance(mult, bool):
            self.mult_list: List[bool] = [mult for _ in range(
                self._num_stateless_modules)]
        else:
            self.mult_list: List[bool] = mult
            if self.mult_list[-1] == True:
                print('Warning: mult set to true in the final layer, ' +
                      'overriding it with False (out_dims must be 1)')
        self.mult_list[-1] = False

        if any(self.mult_list):
            self.mult_nonlinearity = get_nonlinearity(
                nonlinearity=mult_nonlinearity)

    @property
    def requires_position_ids(self,):
        
        return True
    
    def register_embedding(self, embedding_module: TokenEmbedding):
        ScoringNetwork.register_embedding(
            self=self,
            embedding_module=embedding_module,
            )
        self.parameter_dim_per_submodule: List[int] = []
        
        stateless_module = self.stateless_modules_list[0]
        print(stateless_module)
        default_output_features_mult = 1
        if self.mult_list[0] and self.residual_list[0]:
            default_output_features_mult = 2
        stateless_module.instantiate_and_setup_ops(
            input_features=self.input_embedding_dim,
            output_features=None,
            preceding_module=None,
            default_output_features_mult=default_output_features_mult,
            )
        self.parameter_dim_per_submodule.append(
            stateless_module.total_base_parameter_dims)
        
        for i, next_stateless_module in enumerate(
            self.stateless_modules_list[1:]):
            
            
            
            if self.mult_list[i] and self.residual_list[i]:
                previous_input_dim = stateless_module.output_features // 2
                assert stateless_module.output_features % 2 == 0
                assert previous_input_dim == stateless_module.input_features
            elif self.mult_list[i] or self.residual_list[i]:
                previous_input_dim = stateless_module.output_features
                
                
                
                
                assert previous_input_dim == stateless_module.input_features
            else:
                previous_input_dim = stateless_module.output_features

            default_output_features_mult = 1
            if self.mult_list[i+1] and self.residual_list[i+1]:
                default_output_features_mult = 2
            next_stateless_module.instantiate_and_setup_ops(
                input_features=previous_input_dim,
                output_features=None,
                preceding_module=None,
                default_output_features_mult=default_output_features_mult,
                )
            stateless_module = next_stateless_module
            self.parameter_dim_per_submodule.append(
                stateless_module.total_base_parameter_dims)
            
        
        assert stateless_module.output_features == 1

        self.registered_stateless_modules = nn.ModuleList(
            self.stateless_modules_list)
        
        self.total_base_parameters = sum(self.parameter_dim_per_submodule)
        
        
        

        
    
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
        analyze=False,
        
        **kwargs,
        ) -> torch.Tensor:
        '''Produces score for each KV cache token embedding'''

        
        
        

        if not self.requires_recomputation:
            token_embeddings = token_embeddings[..., -num_new_tokens:, :]
        
        
        token_embeddings_shape = token_embeddings.shape
        batch_size, n_heads = token_embeddings_shape[:2]
        parallel_operations = batch_size
        

        if not self.is_reduced_input:
            n_non_reduced_inputs, n_tokens = token_embeddings_shape[-3:-1]
            position_ids = position_ids.unsqueeze(dim=-2).expand(
                batch_size, n_heads, n_non_reduced_inputs, n_tokens)

        if self.shared and self.per_head:
            parallel_operations = parallel_operations*n_heads
            
            
            n_parallel_dimensions = 2
            
        else:
            
            n_parallel_dimensions = 1
            

        reshaped_parameters = parameters.view(
            parallel_operations, self.net_param_size())

        parameters_per_submodule = torch.split_with_sizes(
            reshaped_parameters,
            split_sizes=self.parameter_dim_per_submodule,
            dim=-1,
            )
        
        for stateless_module, p in zip(
            self.stateless_modules_list, parameters_per_submodule):
            stateless_module.load_parameters(
                parameters=p,
                parallel_operations=parallel_operations,
                )

        current_output = token_embeddings
        for stateless_module, res_connection, mult_interaction in zip(
            self.stateless_modules_list, self.residual_list, self.mult_list):
            next_output = stateless_module(
                inputs=current_output,
                attn_weights=attn_weights,
                attn_mask=attn_mask,
                position_ids=position_ids,
                n_parallel_dimensions=n_parallel_dimensions,
                **kwargs,
                )
            if res_connection and mult_interaction:
                
                
                
                res_component, mult_component = torch.chunk(
                    next_output, chunks=2, dim=-1)
                mult_component = self.mult_nonlinearity(mult_component)
                
                

                current_output = current_output*(
                    1 + mult_component) + res_component
                
                
                
                
                
            elif res_connection:
                
                
                
                
                current_output = current_output + next_output
            elif mult_interaction:
                next_output = self.mult_nonlinearity(next_output)
                current_output = current_output*(1 + next_output)
            else:
                current_output = next_output
            
            

        
        scores = current_output.squeeze_(dim=-1)

        if not self.requires_recomputation:
            if not new_sequences:
                
                past_scores: torch.Tensor = self.past_scores[layer_id]
                scores = torch.concat([past_scores, scores], dim=-1)
            if not analyze:
                self.past_scores[layer_id] = scores
        
        
        if analyze:
            pre_process_scores = torch.clone(scores)
            self._analysis_dict['pre_process_scores'] = pre_process_scores
            
        scores = self.process_output(
            layer_id=layer_id,
            ema_coeff=self.ema_coeff,
            num_new_tokens=num_new_tokens,
            new_sequences=new_sequences,
            component_output=scores,
            attn_mask=attn_mask,
            analyze=analyze,
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
        return self.total_base_parameters
    

class TCNScoring(ScoringNetwork):
    '''Temporal Convolutional Network scoring layer, producing score as the NN output combination of the
       embeddings.'''
    def __init__(
            self,
            per_layer: bool, 
            per_head: bool, 
            shared: bool,
            output_params: ComponentOutputParams,
            initializer: numbers.Number = 0,
            kernel_size: Tuple[int, int] = (3, 3),
            stride: Tuple[int, int] = (1, 1),
            out_emb_reduction_mode: str = None, 
            tcn_out_channels: List[int] = [1],
            dilation: Tuple[int, int] = [1, 1],
            dtype: Optional[Union[str, torch.dtype]] = None,
            
            ):
        
        ScoringNetwork.__init__(
            self,
            per_layer=per_layer,
            per_head=per_head,
            shared=shared,
            output_params=output_params,
            buffer_names=['past_scores'],
            initializer=initializer,
            dtype=dtype,
            )

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.out_emb_reduction_mode = out_emb_reduction_mode

        self.tcn_out_channels = tcn_out_channels
        
        
        receptive_field = [None, None]
        receptive_field[0] = (kernel_size[0] - 1) * dilation[0] + 1
        receptive_field[1] = (kernel_size[1] - 1) * dilation[1] + 1
        assert receptive_field[1] % 2 == 1 

        print(f"receptive filed of tcn net is {receptive_field[1] ** len(tcn_out_channels)}")

        padding = [None, None]
        padding[0] = int(receptive_field[0] // 2)
        padding[1] = int(receptive_field[1] // 2)
        self.padding = padding

    def register_embedding(self, embedding_module: TokenEmbedding):
        ScoringNetwork.register_embedding(
            self=self,
            embedding_module=embedding_module,
            )
        self.tcn_net = nn.Sequential()
        for i, tcn_out in enumerate(self.tcn_out_channels):
            self.tcn_net.add_module(
                f"layer_{i}",
                nn.Conv2d(
                    in_channels=self.input_embedding_dim if i==0 else self.tcn_out_channels[i-1],
                    out_channels=tcn_out,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    bias=True,
                )
            )
            if i < len(self.tcn_out_channels) - 1:
                self.tcn_net.add_module(
                    f"relu_{i}",
                    nn.ReLU()
                )
        
        self.tcn_num_params = sum(p.numel() for p in self.tcn_net.parameters())
        self.tcn_net.eval()

        if self.out_emb_reduction_mode == 'linear':
            assert self.tcn_out_channels != 1, f"""
            Increase out_channels > 1 if using Linear out_emb_reduction_mode
            """
            self.out_emb_linear = nn.Linear(in_features=self.tcn_out_channels,
                                         out_features=1, bias=True)
            self.out_emb_linear.eval()
            self.out_num_params = sum(p.numel() for p in self.out_emb_linear.parameters())

    def parameters_to_weight_dict(self, parameters):
        tcn_net_dict = {}
        tcn_parameters = parameters[:self.tcn_num_params]

        for k in self.tcn_net.state_dict():
            params_len = list(self.tcn_net.state_dict()[k].view(-1).shape)[0]
            k_params = tcn_parameters[:params_len]
            k_params = k_params.view(self.tcn_net.state_dict()[k].shape)
            tcn_net_dict[k] = k_params 
            tcn_parameters = tcn_parameters[params_len:]
        assert len(tcn_parameters) == 0

        out_linear_dict = {}
        if self.out_emb_reduction_mode == 'linear':
            out_parameters = parameters[-self.out_num_params:]
            out_linear_dict['weight'] = out_parameters[:, :-1].view(self.out_emb_linear.weight.shape)
            out_linear_dict['bias'] = out_parameters[:, -1]

        return tcn_net_dict, out_linear_dict

    def get_tokens_score(
        self,
        layer_id,
        parameters: torch.Tensor,
        token_embeddings: torch.Tensor,
        new_sequences,
        num_new_tokens,
        attn_weights=None,
        attn_mask=None,
        position_ids=None,
        **kwargs,
        ) -> torch.Tensor:
        '''Produces score for each KV cache token embedding'''
        assert parameters.shape[0] == 1, "TCN doesn't support population batch parallel."
        parameters.squeeze_()

        if self._custom_dtype is not None:
            parameters = parameters.to(dtype=self.ptdtype)
            token_embeddings = token_embeddings.to(dtype=self.ptdtype)

        if not self.requires_recomputation:
            token_embeddings = token_embeddings[..., -num_new_tokens:, :]

        
        tcn_net_dict, out_linear_dict = self.parameters_to_weight_dict(parameters)
        self.tcn_net.load_state_dict(tcn_net_dict, strict=True)
        if self.out_emb_reduction_mode == 'linear':
            self.out_emb_linear.load_state_dict(out_linear_dict, strict=True)
       
        
        batch_size, n_heads, hop_len, seq_len, emb_dim = token_embeddings.shape        
        token_embeddings = token_embeddings.view(-1, hop_len, seq_len, emb_dim)
        
        

        token_embeddings = token_embeddings.permute(0, 3, 1, 2)
        
        

        scores: torch.Tensor = self.tcn_net(token_embeddings)
        
        
        _, _, hop_len_out, seq_len_out = scores.shape

        assert seq_len_out == seq_len, f"""
            The output seq_len after Convolution Network should be the same as the input seq_len,
            but get output {seq_len_out} != input {seq_len}.
            Adjust the padding, kernel size, stride of the Convolution layer.
            """

        if self.out_emb_reduction_mode == 'linear':
            scores = self.out_emb_linear(F.relu(scores.permute(0, 2, 3, 1)))
            
            
            

        scores = scores.view(batch_size, n_heads, hop_len_out, seq_len_out)
        
        
        

        if not self.requires_recomputation:
            if not new_sequences:
                
                past_scores: torch.Tensor = self.past_scores[layer_id]
                scores = torch.concat([past_scores, scores], dim=-1)
            self.past_scores[layer_id] = scores
            
        
        
        scores = self.process_output(
            layer_id=layer_id,
            ema_coeff=self.ema_coeff,
            num_new_tokens=num_new_tokens,
            new_sequences=new_sequences,
            component_output=scores,
            attn_mask=attn_mask,
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
        num_params = self.tcn_num_params
        if self.out_emb_reduction_mode == 'linear':
            num_params += self.out_num_params
        return num_params
    
 