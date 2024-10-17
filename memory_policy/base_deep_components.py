import os
import pdb
import copy
import math
import numbers
import inspect
import numpy as np 
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, List, Callable

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
from .shared import RegistrationCompatible, SynchronizableBufferStorage
from utils import (
    compute_masked_statistics, merge_statistics, 
    compute_masked_statistics_with_var, merge_statistics_from_var,
    safe_tensor_print, # for debugging
    )


def get_property_list_from_elements(property, elements):
    return [getattr(element, property) for element in elements]

def true_for_any(property, elements):
    return any(get_property_list_from_elements(
        property=property,
        elements=elements))

def true_for_all(property, elements):
    return all(get_property_list_from_elements(
        property=property,
        elements=elements))

def get_matching_value(property, elements):
    values = get_property_list_from_elements(
        property=property,
        elements=elements)
    assert len(set(values)) == 1
    return values[0]

def call_for_all(fn_name, elements, *args, **kwargs):
    fn_objects = get_property_list_from_elements(
        property=fn_name,
        elements=elements)
    return_values = [fn_object_i(*args, **kwargs) for fn_object_i in fn_objects]
    return return_values

def reconstruct_strided_causal_mask(
        n_q, # num new elements to mask
        n_k, # num all elements (including unmasked)
        stride, # num scores = n_k // stride
        device,
        dtype,
        ):
    
    n_out = n_q//stride

    causal_mask = torch.ones((n_out, n_out), dtype=dtype, device=device)
    causal_mask = torch.tril(causal_mask, diagonal=0)
    causal_mask = torch.repeat_interleave(causal_mask, repeats=stride, dim=-1)

    causal_mask = F.pad(causal_mask, (n_k-n_q, 0), value=1)

    # n_k x n_num_new_elements
    return causal_mask

def reduce_ema_values(
        outputs, 
        ema_coeff,
        # num_new_outputs,
        num_new_tokens,
        reduction_dim, # only supports -2 and -3 for now
        needs_causal_mask,
        ):
    '''Reduces num_new_tokens//stride scores via a rolling EMA based on the
        recency of each score'''
    device = outputs.device

    if reduction_dim == -2:
        *batch_ds, n_heads, num_new_outputs, num_all_tokens = outputs.shape
    elif reduction_dim == -3:
        *batch_ds, n_heads, num_new_outputs, num_all_tokens, embedding_dim = (
            outputs.shape)
    else:
        raise NotImplementedError
    
    residual = num_new_tokens % num_new_outputs
    if residual > 0:
        raise NotADirectoryError
    stride = num_new_tokens // num_new_outputs
    discount_vector_exponents = torch.arange(
        start=num_new_outputs-1, end=-1, step=-1,
        device=device,)# dtype=dtype)
    
    accumulated_ema_coeff = ema_coeff**stride
    # from ema^(num_new_tokens-1) to ema^0
    discount_vector = torch.pow(
        accumulated_ema_coeff, discount_vector_exponents)

    discount_vector = discount_vector*(1-accumulated_ema_coeff)

    
    if reduction_dim == -2:
        discount_vector = discount_vector.view(num_new_outputs, 1)
    else:
        discount_vector = discount_vector.view(num_new_outputs, 1, 1)
    
    if needs_causal_mask:
        n_q, n_k = num_new_outputs*stride, num_all_tokens
        # n_new_outputs, n_k
        mask = reconstruct_strided_causal_mask(
            n_q=n_q, # num new elements to mask
            n_k=n_k, # num all elements (including unmasked)
            stride=stride, # num scores = n_k // stride
            device=device,
            dtype=discount_vector.dtype,)
        if reduction_dim == -3:
            mask = mask.view(num_new_outputs, n_k, 1)
        discount_vector = discount_vector*mask
    # bs x n_heads x n_k (x embedding_dim)
    reduced_attn_weights = (discount_vector*outputs).sum(dim=reduction_dim)
    
    return reduced_attn_weights

def wrap_torch_initializer(
        initializer_fn_: Callable, *args, **kwargs) -> Callable:
    def wrapped_initializer(shape):
        tensor = torch.zeros(size=shape)
        initializer_fn_(tensor, *args, **kwargs)
        return tensor
    return wrapped_initializer

@dataclass
class EMAParams:
    coeff: float
    learned: bool
    # how often to to reduce scores e.g., 128 -> every 128 tokens
    # if None/-1 reduce adaptively based on number new tokens (minimizes 
    # computation, can be different each iteration)
    reduction_stride: Optional[int] = None

# ..., T x num_all_tokens x emb_dim-
@dataclass
class ComponentOutputParams:
    requires_recomputation: bool
    # given new_seq_len > 1 new tokens, we have (new_seq_len - 1) previous
    # 'states' for which we can compute intermediate scores
    # should be
    # -  None/'none' (no reduction)
    # - 'last' (only consider final state with all latest tokens)
    # - 'mean' (avg. across scores)
    # - 'ema' (exp. moving avg., requires ema_params)
    reduction_mode: Optional[str] = None 
    ema_params: Optional[EMAParams] = None
    output_past_non_reduced_history: bool = False
    max_non_reduced_history_len: Optional[int] = None
    # whether to use global statistics to normalize the component output to zero
    # mean and unit variance
    online_output_normalization: bool = False
    update_norm_during_training: bool = False
    update_norm_during_eval: bool = True



class DeepMemoryPolicyComponent(
    nn.Module,
    RegistrationCompatible,
    SynchronizableBufferStorage,
    abc.ABC,
    ):
    '''Component for the deep memory policies. For convenience should be
       stateless and take an input the necessary parameters'''
    def __init__(self, 
                 main_method_name: str,
                 per_layer: bool,
                 per_head: bool, 
                 shared: bool,
                 output_params: ComponentOutputParams,
                 buffer_names: Optional[List[str]],
                 buffers_to_merge_keys: Optional[List[str]] = None,
                 initializer: Union[
                     numbers.Number, np.array, torch.Tensor, Callable] = 0,
                 has_embedding_dim: bool =False,
                 dtype: Optional[Union[str, torch.dtype]] = None,
                 ):
        nn.Module.__init__(self,)
        self.per_layer = per_layer
        self.per_head = per_head
        self.shared = shared
        self.buffer_names = buffer_names
        self._buffers_to_merge_names = []
        self._analysis_dict = {}
        self.initializer = initializer
        self.has_embedding_dim = has_embedding_dim
        self.aux_params_init = []
        if not hasattr(self, '_num_aux_params'):
            self._num_aux_params = 0
        self._has_aux_param = self._num_aux_params > 0
        self.aux_params_getter_names: dict = {}
        self.store_output_params(output_params=output_params)
        self.main_method_name = main_method_name
        assert hasattr(self, self.main_method_name)

        if not hasattr(self, '_has_buffers_to_merge'):
            SynchronizableBufferStorage.__init__(
                self=self,
                buffers_to_merge=self._buffers_to_merge_names,
                sub_buffer_storages=[],
                )
        
        self.setup_dtype(dtype=dtype)

    def get_analysis_dict(self,):
        to_return = {k: v for k, v in self._analysis_dict.items()}
        self._analysis_dict = {}
        return to_return
    
    def setup_dtype(self, dtype: Optional[Union[str, torch.dtype]] = None):
        self._custom_dtype = False
        self.dtype = dtype
        self.ptdtype = None
        if dtype is not None:
            self._custom_dtype = True
            if isinstance(dtype, torch.dtype):
                self.ptdtype = dtype
            else:
                self.ptdtype = {
                    'float32': torch.float32,
                    'bfloat16': torch.bfloat16,
                    'float16': torch.float16,
                    }[self.dtype]
    
    def override_ema_coeff(self, new_ema):
        if self.ema_output:
            assert not self.learned_ema_coeff
            self.ema_coeff = new_ema


    def store_output_params(self, output_params: ComponentOutputParams):
        self._requires_recomputation = output_params.requires_recomputation
        self.reduction_mode = output_params.reduction_mode
        self.ema_output = False
        if self.reduction_mode is not None:
            self.reduction_mode = self.reduction_mode.lower()
            # implemented options, so far
            assert self.reduction_mode in ['none', 'last', 'ema']
            self.ema_output = self.reduction_mode == 'ema'
            if self.reduction_mode == 'none':
                self.reduction_mode = None

            self.reduction_dim = -2
            if self.has_embedding_dim:
                self.reduction_dim = -3
        self.output_past_non_reduced_history = (
            output_params.output_past_non_reduced_history)
        if self.output_past_non_reduced_history:
            assert self.reduction_mode is None and not self.reduced_output
            self.past_outputs_buffer: List[Optional[torch.Tensor]]
            self.buffer_names += ['past_outputs_buffer']

        self.max_non_reduced_history_len = (
            output_params.max_non_reduced_history_len)
        self.limit_past_history_size = False
        if self.max_non_reduced_history_len is not None:
            assert self.output_past_non_reduced_history
            self.limit_past_history_size = True
        self.learned_ema_coeff = False
        self.ema_reduction_stride = None
        self.ema_coeff = None
        if self.ema_output:
            ema_params = output_params.ema_params
            self.ema_reduction_stride = ema_params.reduction_stride
            self.ema_coeff = ema_params.coeff
            self.learned_ema_coeff = ema_params.learned

            self.ema_output_buffer: List[Optional[torch.Tensor]]
            self.buffer_names += ['ema_output_buffer']
            if self.learned_ema_coeff:
                self.setup_aux_param(
                    param_name='ema_coeff',
                    initial_value=self.ema_coeff,
                    scaling=(0, 1))
        self.online_output_normalization = (
            output_params.online_output_normalization)
        self.update_norm_during_eval = (
            output_params.update_norm_during_eval)
        self.update_norm_during_training = (
            output_params.update_norm_during_training)
        
        if self.online_output_normalization:
            assert (self.update_norm_during_training or 
                    self.update_norm_during_eval)
        
        
        self._empty_norm_buffers = True
        self._empty_new_norm_buffers = True
        if output_params.online_output_normalization:
            # online_new contains only for the current iterations
            self._buffers_to_merge_names += [
                'online_new_output_mean',
                'online_new_output_var',
                'online_new_output_num',
                'output_mean',
                'output_var',
                'output_num',
                ]
    
    def register_sub_buffers_to_merge(self, sub_buffer_storages: list):
        SynchronizableBufferStorage.register_sub_buffers_to_merge(
            self=self, sub_buffer_storages=sub_buffer_storages)
        sub_buffer_storages_norm = [
            b for b in sub_buffer_storages if b.online_output_normalization
            ]
        
        if len(sub_buffer_storages_norm) > 0:
            if self.online_output_normalization:
                sub_buffer_storages_norm += [self]

    
    def _merge_own_buffers(
            self,
            buffers_to_merge: List[List[torch.Tensor]],
            ) -> List[torch.Tensor]:
        merged_buffers = []
        if self.online_output_normalization:
            assert len(buffers_to_merge) == 6
            (new_mean_per_ddp, new_var_per_ddp, new_num_per_ddp,
             mean_per_ddp, var_per_ddp, num_per_ddp) = buffers_to_merge
            
            # merge own mean, var, num (which should already include current
            # estimate) with other ddp's new mean, var, and num
            merged_mean, merged_var, merged_num = (
                mean_per_ddp[0], var_per_ddp[0], num_per_ddp[0])
            

            
            for other_mean, other_var, other_num in zip(
                new_mean_per_ddp, new_var_per_ddp, new_num_per_ddp):

                merged_mean, merged_var, merged_num = (
                    merge_statistics_from_var(
                        mean_a=merged_mean, 
                        variance_a=merged_var, 
                        num_a=merged_num,
                        mean_b=other_mean,
                        variance_b=other_var,
                        num_b=other_num,
                    ))
                
                
                
                
            new_merged_mean = torch.zeros_like(merged_mean)
            new_merged_var = torch.zeros_like(merged_var)
            new_merged_num = torch.zeros_like(merged_num)
            merged_buffers += [
                new_merged_mean,
                new_merged_var,
                new_merged_num,
                merged_mean,
                merged_var,
                merged_num,
            ]
        else:
            assert len(buffers_to_merge) == 0  
        return merged_buffers  
    
    def _self_merge_own_buffers(self,) -> List[torch.Tensor]:
        buffers_to_merge = self.get_buffers_list()
        merged_buffers = []
        if self.online_output_normalization:
            assert len(buffers_to_merge) == 6
            (new_mean_per_ddp, new_var_per_ddp, new_num_per_ddp,
             merged_mean, merged_var, merged_num) = buffers_to_merge
            
            # merge own mean, var, num (which should already include current
            # estimate) with other ddp's new mean, var, and num
            merged_mean, merged_var, merged_num = (
                merge_statistics_from_var(
                    mean_a=merged_mean, 
                    variance_a=merged_var, 
                    num_a=merged_num,
                    mean_b=new_mean_per_ddp,
                    variance_b=new_var_per_ddp,
                    num_b=new_num_per_ddp,
                ))
            
            new_merged_mean = torch.zeros_like(merged_mean)
            new_merged_var = torch.zeros_like(merged_var)
            new_merged_num = torch.zeros_like(merged_num)
            merged_buffers += [
                new_merged_mean,
                new_merged_var,
                new_merged_num,
                merged_mean,
                merged_var,
                merged_num,
            ]
        else:
            assert len(buffers_to_merge) == 0
        return merged_buffers
    
    def get_output_normalization_dims(self,):
        '''Input is bs x n_heads (x T) x n_all_tokens x emb_dim'''
        if self.reduced_output:
            return [0, 1, 2]
        else:
            return [0, 1, 2, 3]

    def setup_aux_param(self, param_name, initial_value, scaling):
        self._has_aux_param = True
        main_scaling = self.get_param_scaling()
        assert main_scaling == 'linear'
        curr_aux_counter = copy.copy(self._num_aux_params)
        exp_scaling = False
        bounded = False
        bounded_min = 0
        bounded_range = 1
        print(f'Initializing auxiliary parameter: {param_name}')
        if isinstance(scaling, str):
            assert scaling in ['linear', 'exp']
            if scaling == 'exp':
                exp_scaling = True
                assert initial_value > 1e-7
                initial_value = np.log(initial_value)
        else:
            min_max_list = list(scaling)
            assert len(min_max_list) == 2
            assert min_max_list[0] < initial_value
            assert min_max_list[1] > initial_value
            bounded_min = min_max_list[0]
            bounded_range = min_max_list[1] - min_max_list[0]
            bounded = True
            # in [0, 1]
            norm_initial_value = (initial_value - bounded_min)/bounded_range
            initial_value = np.log(norm_initial_value) - np.log(
                1 - norm_initial_value)
        self.aux_params_init.append(initial_value)
        self._num_aux_params += 1

        def getter(aux_params):
            v = aux_params[..., [curr_aux_counter]]
            if exp_scaling:
                v = torch.exp(v)
            elif bounded:
                v = torch.sigmoid(v)*bounded_range + bounded_min
            return v
        
        getter_name = 'get_' + param_name
        setattr(self, getter_name, getter)
        self.aux_params_getter_names[param_name] = getter_name

    @property
    def reduced_output(self,):
        if self.reduction_mode is None:
            return False
        else:
            return True
        
    @property
    def requires_attn_scores(self,):
        # regarding update cache operation
        return False
    
    @property
    def requires_queries(self,):
        # regarding update cache operation
        return False
    
    @property
    def requires_position_ids(self,):
        return False
    
    @property
    def is_diversity_based(self,):
        return False
    
    def load_buffers_dict(self, buffers_dict, updating_only_new=False):
        SynchronizableBufferStorage.load_buffers_dict(
            self=self,
            buffers_dict=buffers_dict,
            )
        if len(buffers_dict) == 0 or updating_only_new:
            return
        else:
            # start using old rather than new stats
            self._empty_norm_buffers = False
            self._empty_new_norm_buffers = False

    
    @property
    def requires_recomputation(self,):
        '''Whether requires recomputation every time another token in the KV 
           cache changes (e.g., attention-based).
           If false output will only include num_new_tokens final tokens.'''
        return self._requires_recomputation

    def net_param_size(self,) -> int:
        return 0
    
    def aux_param_size(self,) -> int:
        return self._num_aux_params
        
    def get_param_size(self,) -> int:
        return self.net_param_size() + self.aux_param_size()
    
    def get_net_param_initial(self,) -> np.ndarray:
        net_param_size = self.net_param_size()
        if isinstance(self.initializer, numbers.Number):
            initial_values = self.initializer*np.ones([net_param_size])
        elif isinstance(self.initializer, Callable):
            initial_values = self.initializer([net_param_size])
        else:
            initial_values = self.initializer
        if isinstance(initial_values, torch.Tensor):
            initial_values = initial_values.cpu().numpy()
        assert isinstance(initial_values, np.ndarray)
        assert initial_values.shape[-1] == net_param_size
        return initial_values
    
    def get_aux_param_initial(self,) -> np.ndarray:
        return np.array(self.aux_params_init)
    
    def get_param_initial(self,) -> np.ndarray:
        params_init = self.get_net_param_initial()
        if self._has_aux_param:
            aux_params_init = self.get_aux_param_initial()
            params_init = np.concatenate(
                [params_init, aux_params_init], axis=-1)
        return params_init
    
    def split_net_and_aux_params(self, parameters: Optional[torch.Tensor]):
        if self._has_aux_param:
            return torch.split_with_sizes(
                input=parameters, 
                split_sizes=[self.net_param_size(), self.aux_param_size()],
                dim=-1)
        else:
            return parameters, None
    
    def get_param_scaling(self,) -> Optional[Union[str, Tuple[float, float]]]:
        return 'linear'
    
    def get_param_scaling_dims(
            self,) -> Tuple[int, int, int]:
        # return number of linear, exp., and bounded parameters
        # for now, not used/implemented
        raise NotImplementedError
    
    def finalize_registration(self,):
        self.initialize_buffers()
    
    def initialize_buffers(self,):
        for buffer_name in self.buffer_names:
            empty_buffer = [None for _ in range(self.num_memory_layers)]
            setattr(self, buffer_name, empty_buffer)
    
    def filter_buffer_values(
            self,
            layer_id: int,
            # bs x n_heads x num_filtered_tokens
            retained_idxs: torch.Tensor,
            ):
        '''Filters all buffers to only include retained idxs'''
        if self.ema_output:
            if self.has_embedding_dim:
                # expand across the ouput (embedding) dim. (if present)
                retained_idxs = retained_idxs.unsqueeze(-1).expand(
                    -1, -1, -1, self.get_embedding_dim())
            ema_output: torch.Tensor = self.ema_output_buffer[layer_id]
                
            self.ema_output_buffer[layer_id] = torch.gather(
                input=ema_output, dim=2, index=retained_idxs)
        elif self.output_past_non_reduced_history:
            # bs x num_heads x history_len x num_all_tokens x emb_dim
            current_output_history = self.past_outputs_buffer[layer_id]
            if self.has_embedding_dim:
                # expand across the ouput (embedding) dim. (if present)
                batch_size, n_heads, history_len, n_all_tokens, emb_dim = (
                    current_output_history.shape)
                retained_idxs = retained_idxs.unsqueeze(-2).unsqueeze(
                    -1).expand(
                        -1, -1, history_len, -1, self.get_embedding_dim())
            else:
                batch_size, n_heads, history_len, n_all_tokens = (
                    current_output_history.shape)
                retained_idxs = retained_idxs.unsqueeze(-2).expand(
                    -1, -1, history_len, -1)
                                
            self.past_outputs_buffer[layer_id] = torch.gather(
                input=current_output_history, dim=3, index=retained_idxs)

    def process_output_ema(
            self,
            layer_id,
            ema_coeff,
            num_new_tokens,
            new_sequences,
            component_output,
            analyze=False,
            ):

        if self.learned_ema_coeff:
            # ema_coeff could be different for each head
            bs, unique_heads = ema_coeff.shape
            if self.has_embedding_dim:
                ema_coeff = ema_coeff.view(bs, unique_heads, 1, 1, 1)
            else:
                ema_coeff = ema_coeff.view(bs, unique_heads, 1, 1)
        # bs x unique_heads x num_all_tokens (x embedding_dim)
        reduced_output = reduce_ema_values(
            outputs=component_output,
            ema_coeff=ema_coeff,
            num_new_tokens=num_new_tokens,
            reduction_dim=self.reduction_dim,
            needs_causal_mask=True,
            )
        if not new_sequences:
            # bs x unique_heads x num_old_tokens (x embedding_dim)
            old_tokens_output: torch.Tensor = self.ema_output_buffer[
                layer_id]
            scaled_ema = ema_coeff**num_new_tokens
            if self.learned_ema_coeff:
                # bs x unique_heads x 1 (x embedding_dim)
                scaled_ema = scaled_ema.squeeze(-1)
            # allow broadcasting
            scaled_old_tokens_output = old_tokens_output*scaled_ema
            
            if self.has_embedding_dim:
                
                
                
                reduced_output[..., :-num_new_tokens, :] = (
                    reduced_output[..., :-num_new_tokens, :] + 
                    scaled_old_tokens_output)
            else:
                reduced_output[..., :-num_new_tokens] = (
                    reduced_output[..., :-num_new_tokens] + 
                    scaled_old_tokens_output)
        if not analyze:
            self.ema_output_buffer[layer_id] = reduced_output
        return reduced_output


    def process_output(
            self,
            layer_id,
            ema_coeff,
            num_new_tokens,
            new_sequences,
            component_output,
            aux_params: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            ema_reduction_over: Optional[float] = None,
            analyze: bool = False,
            **kwargs,
            ):
        if self.ema_output:
            if ema_reduction_over is not None:
                
                ema_coeff = ema_reduction_over
            elif self.learned_ema_coeff:
                ema_coeff = self.get_ema_coeff(aux_params=aux_params)

            reduced_component_output = self.process_output_ema(
                layer_id=layer_id,
                ema_coeff=ema_coeff,
                num_new_tokens=num_new_tokens,
                new_sequences=new_sequences,
                component_output=component_output,
                analyze=analyze,
                )
        elif self.reduction_mode is None:
            reduced_component_output = component_output
        elif self.reduced_mode == 'last':
            if self.has_embedding_dim:
                reduced_component_output = component_output[..., -1, :]
            else:
                reduced_component_output = component_output[..., -1]
        elif self.reduced_mode == 'mean':
            if self.has_embedding_dim:
                reduced_component_output = component_output.mean(-2)
            else:
                reduced_component_output = component_output.mean(-1)
        elif not self.reduced_output:
            raise NotImplementedError
        
        if self.online_output_normalization:
            if self.are_sync_buffers_frozen():
                update_norm = False
            else:
                if self._is_in_training_mode:
                    if self.update_norm_during_training:
                        update_norm = True
                    else:
                        update_norm = False
                else:
                    if self.update_norm_during_eval:
                        update_norm = True
                    else:
                        update_norm = False
                
            if update_norm:
                if self._empty_new_norm_buffers:
                    self._empty_new_norm_buffers = False
                output_shape = reduced_component_output.shape
                bs = output_shape[0]
                if self.has_embedding_dim:
                    num_all_tokens = output_shape[-2]
                else:
                    num_all_tokens = output_shape[-1]
                
                attn_mask = attn_mask.view(
                    bs, 1, -1)[..., -num_all_tokens:].bool()
                
                if not self.reduced_output:
                    attn_mask = attn_mask.unsqueeze(-2)
                if self.has_embedding_dim:
                    attn_mask = attn_mask.unsqueeze(-1)

                mean, variance, total_num = compute_masked_statistics_with_var(
                    values=reduced_component_output,
                    mask=attn_mask,
                    reduce_dims=self.get_output_normalization_dims(),
                    )

                buffers_dict = self.get_buffers_dict()
                
                mean, variance, total_num = merge_statistics_from_var(
                        mean_a=mean, 
                        variance_a=variance, 
                        num_a=total_num,
                        mean_b=buffers_dict['online_new_output_mean'],
                        variance_b=buffers_dict['online_new_output_var'],
                        num_b=buffers_dict['online_new_output_num'],
                )
                buffers_dict['online_new_output_mean'] = mean
                buffers_dict['online_new_output_var'] = variance
                buffers_dict['online_new_output_num'] = total_num
                
                self.load_buffers_dict(buffers_dict=buffers_dict, 
                                       updating_only_new=True)
                # raise NotImplementedError
            else:
                buffers_dict = self.get_buffers_dict()
                
                

            if self._empty_norm_buffers:

                mean = buffers_dict['online_new_output_mean']
                variance = buffers_dict['online_new_output_var']
            else:
                mean = buffers_dict['output_mean']
                variance = buffers_dict['output_var']
            
            if not (self._empty_norm_buffers and self._empty_new_norm_buffers):
                
                
                reduced_component_output = (
                    reduced_component_output - mean)/torch.clamp_min(
                        torch.sqrt(variance), 1e-7)
                
            else:
                print('WARNING: unable to normalize due to empty buffers')
                
        return reduced_component_output
    
    def get_buffers_dict(self,):
        buffers_dict = SynchronizableBufferStorage.get_buffers_dict(self=self,)
        if self.online_output_normalization:
            if self._empty_norm_buffers and (
                'online_new_output_mean' in buffers_dict):
                if isinstance(buffers_dict['online_new_output_mean'],
                              torch.Tensor) and (
                    not isinstance(buffers_dict['output_mean'], torch.Tensor)):
                    buffers_dict['output_mean'] = torch.zeros_like(
                        buffers_dict['online_new_output_mean'])
                    buffers_dict['output_var'] = torch.zeros_like(
                        buffers_dict['online_new_output_var'])
                    buffers_dict['output_num'] = torch.zeros_like(
                        buffers_dict['online_new_output_num'])
        return buffers_dict
    
    def update_norm_stats(self, layer_id, output, attn_mask):
        valid_output = output[..., :-1]
        if attn_mask is not None:
            valid_output_mask = torch.clone(
                attn_mask[..., :-1])
            if self.div_value:
                valid_output_mask = valid_output_mask.unsqueeze(0)
            valid_output_mask = valid_output_mask.expand_as(
                valid_output)
        else:
            valid_output_mask = torch.ones_like(
                valid_output, dtype=torch.bool)
        masked_valid_distances = torch.where(
            valid_output_mask, valid_output, torch.zeros_like(
                valid_output))
        
        
        mean, variance, total_num = compute_masked_statistics_with_var(
            values=masked_valid_distances, mask=valid_output_mask, 
            reduce_dims=self.div_norm_reduce_dims)
        
        if self.div_norm_recompute:
            return mean, variance, total_num
        
        if self.div_norm_need_reset[layer_id]:
            self.div_norm_need_reset[layer_id] = False
            self.norm_mean_est[layer_id] = mean
            self.norm_summed_var_est[layer_id] = variance
            self.norm_num_samples[layer_id] = total_num
            return mean, variance, total_num



        mean_ab, variance_ab, num_ab = merge_statistics_from_var(
            mean_a=self.norm_mean_est[layer_id], 
            variance_a=self.norm_summed_var_est[layer_id], 
            num_a=self.norm_num_samples[layer_id],

            mean_b=mean,
            variance_b=variance,
            num_b=total_num,
            )
        

        self.norm_mean_est[layer_id] = mean_ab
        self.norm_summed_var_est[layer_id] = variance_ab
        self.norm_num_samples[layer_id] = num_ab
        return mean_ab, variance_ab, num_ab

    
                                
    def reset_param_stats(self,):
        pass

    def latest_stats(self,) -> dict:
        stats = dict()
        return stats
    
    def get_aux_params_stats(self, aux_params: Optional[torch.Tensor]):
        stats = dict()
        if self._has_aux_param:
            prefix = 'aux_params/'
            for p_name, getter_fn_name in self.aux_params_getter_names.items():
                getter_fn = getattr(self, getter_fn_name)
                stats[prefix + p_name] = getter_fn(
                    aux_params=aux_params).mean().item()
        return stats

    def get_net_params_stats(self, parameters: Optional[torch.Tensor]):
        stats = {}
        prefix = 'net_params/'
        if self.net_param_size() > 0:
            mean = parameters.mean()
            stats[prefix + 'mean'] = mean.item()
            stats[prefix + 'std'] = parameters.std(dim=-1).mean().item()
            stats[prefix + 'max'] = parameters.max(dim=-1)[0].mean().item()
            stats[prefix + 'min'] = parameters.min(dim=-1)[0].mean().item()
            stats[prefix + 'above_avg'] = (
                parameters > mean).float().mean().item()
        return stats
    
    def get_param_stats(self, parameters, reset=True) -> dict:
        parameters, aux_params = self.split_net_and_aux_params(
            parameters=parameters)
        stats = self.get_net_params_stats(parameters=parameters)
        stats.update(self.get_aux_params_stats(aux_params=aux_params))
        return stats
    
    # @abc.abstractmethod
    def load_parameters(
            self,
            parameters: torch.Tensor,
            # should match first dimension of parameters
            parallel_operations: Optional[int] = None,
            ):
        raise NotImplementedError


    def forward(self, *args, **kwargs):
        return getattr(self, self.main_method_name)(*args, **kwargs)

class TokenEmbedding(DeepMemoryPolicyComponent, abc.ABC):
    def __init__(
            self,
            per_layer: bool,
            per_head: bool,
            shared: bool,
            output_params: ComponentOutputParams, 
            buffer_names: Optional[List[str]],
            initializer: Union[float, np.ndarray, torch.Tensor, callable]=0,
            dtype: Optional[Union[str, torch.dtype]] = None
            ):
        DeepMemoryPolicyComponent.__init__(
            self,
            main_method_name='get_tokens_embedding',
            per_layer=per_layer,
            per_head=per_head,
            shared=shared,
            output_params=output_params,
            buffer_names=buffer_names,
            initializer=initializer,
            has_embedding_dim=True,
            dtype=dtype,
            )

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
        '''Builds a tensor representation for each KV cache token'''
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_embedding_dim(self,) -> int:
        raise NotImplementedError
    

class JointEmbeddings(TokenEmbedding):
    def __init__(
            self,
            embeddings_list: List[Union[TokenEmbedding, DictConfig]],
            joining_strategy: str = 'concat',
            output_params: Optional[ComponentOutputParams] = None,
            ):
        self.joining_strategy = joining_strategy.lower()
        assert self.joining_strategy in ['concat']
        in_embeddings_list: List[TokenEmbedding] = []
        for token_embedding in embeddings_list:
            if isinstance(token_embedding, DictConfig):
                token_embedding: TokenEmbedding = hydra.utils.instantiate(
                    token_embedding)
            else:
                assert isinstance(token_embedding, TokenEmbedding)
            in_embeddings_list.append(token_embedding)
        
        self.requires_recomputation_list = get_property_list_from_elements(
            property='requires_recomputation', elements=in_embeddings_list)
        requires_recomputation = any(self.requires_recomputation_list)

        if output_params is None:
            output_params = ComponentOutputParams(
                requires_recomputation=requires_recomputation)
        else:
            assert (output_params.requires_recomputation ==
                    requires_recomputation)
        per_layer = get_matching_value('per_layer', in_embeddings_list)
        per_head = get_matching_value('per_head', in_embeddings_list)
        shared = get_matching_value('shared', in_embeddings_list)

        self.reduced_outputs_list = get_property_list_from_elements(
            property='reduced_output', elements=in_embeddings_list)
        self._is_reduced_output = all(self.reduced_outputs_list)
        
        param_scaling = None
        _set_param_scaling = False
        self._net_params_list = []
        self._aux_params_list = []
        self._total_params_list = []
        self.embedding_dims_list = []
        for token_embedding in in_embeddings_list:
            net_params: int = token_embedding.net_param_size()
            aux_params: int = token_embedding.aux_param_size()
            num_params: int = token_embedding.get_param_size()
            embedding_dim: int = token_embedding.get_embedding_dim()
            assert net_params + aux_params == num_params
            self._net_params_list.append(net_params)
            self._aux_params_list.append(aux_params)
            self._total_params_list.append(num_params)
            self.embedding_dims_list.append(embedding_dim)
            if num_params > 0:
                if _set_param_scaling:
                    assert token_embedding.get_param_scaling() == param_scaling
                else:
                    _set_param_scaling = True
                    param_scaling = token_embedding.get_param_scaling()

        self._param_scaling = param_scaling

        self._num_net_params = sum(self._total_params_list)
        self._num_aux_params = 0
        self._total_embedding_dim = sum(self.embedding_dims_list)

        self._requires_attn_scores = true_for_any(
            property='requires_attn_scores',elements=in_embeddings_list)
        self._requires_queries = true_for_any(
            property='requires_queries', elements=in_embeddings_list)
        self._requires_position_ids = true_for_any(
            property='requires_position_ids', elements=in_embeddings_list)
        self._is_diversity_based = true_for_any(
            property='is_diversity_based', elements=in_embeddings_list)
        

        TokenEmbedding.__init__(
            self,
            per_layer=per_layer,
            per_head=per_head,
            shared=shared,
            output_params=output_params,
            buffer_names=[],
            initializer=0, 
            )
        
        self.register_sub_buffers_to_merge(
            sub_buffer_storages=in_embeddings_list)
            
        self.embeddings_list = nn.ModuleList(in_embeddings_list)
    
    def override_ema_coeff(self, new_ema):
        DeepMemoryPolicyComponent.override_ema_coeff(self=self, new_ema=new_ema)
        for component in self.embeddings_list:
            component.override_ema_coeff(new_ema=new_ema)

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
        parameters, aux_params = self.split_net_and_aux_params(
            parameters=parameters)
        embeddings_to_join = []
        expanded_dim = 1
        split_parameters = torch.split_with_sizes(
            input=parameters,
            split_sizes=self._total_params_list,
            dim=-1)
        
        for i, (token_embedding, token_embedding_params) in enumerate(
            zip(self.embeddings_list, split_parameters)):

            output_embeddings = token_embedding.get_tokens_embedding(
                layer_id=layer_id,
                parameters=token_embedding_params,
                key_cache=key_cache,
                value_cache=value_cache,
                new_sequences=new_sequences,
                num_new_tokens=num_new_tokens,
                attn_weights=attn_weights,
                attn_mask=attn_mask,
                position_ids=position_ids,
                **kwargs,
            )
            if not self.reduced_outputs_list[i]:
                expanded_dim = output_embeddings.shape[-3]
            else:
                pass
            embeddings_to_join.append(output_embeddings)
        if not self.reduced_output:
            for i, (token_embedding, is_reduced) in enumerate(
                zip(embeddings_to_join, self.reduced_outputs_list)):

                if is_reduced:
                    embeddings_to_join[i] = token_embedding.unsqueeze(
                        dim=-3).expand(-1, -1, expanded_dim, -1, -1)
        
        embeddings_cat = torch.concat(embeddings_to_join, dim=-1)
        embeddings_cat = self.process_output(
                layer_id=layer_id,
                ema_coeff=self.ema_coeff,
                num_new_tokens=num_new_tokens,
                new_sequences=new_sequences,
                component_output=embeddings_cat,
                aux_params=aux_params,
                attn_mask=attn_mask,
                **kwargs,
                )
        return embeddings_cat
    

    def get_embedding_dim(self,) -> int:
        return self._total_embedding_dim

    def net_param_size(self,) -> int:
        return self._num_net_params
    
    def aux_param_size(self,) -> int:
        return self._num_aux_params
    
    def get_param_initial(self,) -> Union[np.ndarray, torch.Tensor]:
        initial_values_list = []
        for i, token_embedding in enumerate(self.embeddings_list):
            initial_values = token_embedding.get_param_initial()
            if isinstance(initial_values, torch.Tensor):
                initial_values = initial_values.cpu().numpy()
            initial_values_list.append(initial_values)
        initial_values_cat = np.concatenate(initial_values_list, axis=-1)
        return initial_values_cat
    
    def get_param_scaling(self,) -> Optional[Union[str, Tuple[float, float]]]:
        return self._param_scaling
    
    def register_new_memory_model(self, config, registration_kwargs):
        call_for_all(
            fn_name='register_new_memory_model',
            elements=self.embeddings_list,
            config=config,
            registration_kwargs=registration_kwargs,
            ) 
        TokenEmbedding.register_new_memory_model(
            self=self,
            config=config,
            registration_kwargs=registration_kwargs,
            )



    def finalize_registration(self,):
        call_for_all(
            fn_name='finalize_registration',
            elements=self.embeddings_list,
            )
        
    def initialize_buffers(self,):
        call_for_all(
            fn_name='initialize_buffers',
            elements=self.embeddings_list,
            ) 
        
    def filter_buffer_values(
            self,
            layer_id: int,
            # bs x n_heads x num_filtered_tokens
            retained_idxs: torch.Tensor,
            ):
        call_for_all(
            fn_name='filter_buffer_values',
            elements=self.embeddings_list,
            layer_id=layer_id,
            retained_idxs=retained_idxs,
            )
    
    def reset_param_stats(self,):
        call_for_all(
            fn_name='reset_param_stats',
            elements=self.embeddings_list,
            )

    def latest_stats(self,):
        call_for_all(
            fn_name='latest_stats',
            elements=self.embeddings_list,
            )
    
    def get_param_stats(self, parameters, reset=True) -> dict:
        stats = dict()
        split_parameters = torch.split_with_sizes(
            input=parameters,
            split_sizes=self._total_params_list,
            dim=-1)
        for i, (token_embedding, token_embedding_params) in enumerate(
            zip(self.embeddings_list, split_parameters)):
            stats_i = token_embedding.get_param_stats(
                parameters=token_embedding_params,
                reset=reset,
                )
            prefix = f'emb_{i}/'
            for k, v in stats_i.items():
                stats[prefix + k] = v
        return stats
    
    def load_parameters(
            self,
            parameters: torch.Tensor,
            # should match first dimension of parameters
            parallel_operations: Optional[int] = None,
            ):
        raise NotImplementedError
    
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
class ScoringNetwork(DeepMemoryPolicyComponent, abc.ABC):
    def __init__(
            self, per_layer: bool, per_head: bool, shared: bool,                 
            output_params: ComponentOutputParams, 
            buffer_names: Optional[List[str]],
            initializer: Union[float, np.ndarray, torch.Tensor, callable]=0,
            dtype: Optional[Union[str, torch.dtype]] = None
            ):
        
        DeepMemoryPolicyComponent.__init__(
            self,
            main_method_name='get_tokens_score',
            per_layer=per_layer,
            per_head=per_head,
            shared=shared,
            output_params=output_params,
            buffer_names=buffer_names,
            initializer=initializer,
            has_embedding_dim=False,
            dtype=dtype,
            )

    @abc.abstractmethod
    def get_tokens_score(
        self,
        layer_id,
        parameters,
        token_embeddings,
        new_sequences,
        num_new_tokens,
        attn_weights=None,
        attn_mask=None,
        position_ids=None,
        # only_score_new_tokens=False,
        **kwargs,
        ) -> torch.Tensor:
        '''Produces score for each KV cache token embedding'''
        raise NotImplementedError
    
    def register_embedding(self, embedding_module: TokenEmbedding):
        self.input_embedding_dim = embedding_module.get_embedding_dim()
        self._requires_recomputation = (
            self._requires_recomputation or
            embedding_module.requires_recomputation)
        self.is_reduced_input = embedding_module.reduced_output
        if self.reduction_mode is not None:
            assert not self.is_reduced_input, ('Specified reduction mode, but' +
                ' scoring network input is not reduced')
            self._is_reduced_output = True
        else:
            self._is_reduced_output = self.is_reduced_input
    
    @property
    def reduced_output(self,):
        return self._is_reduced_output


class SelectionNetwork(DeepMemoryPolicyComponent, abc.ABC):
    def __init__(
        self, per_layer: bool, per_head: bool, shared: bool,                 
        output_params: ComponentOutputParams, 
        buffer_names: Optional[List[str]],
        initializer: Union[float, np.ndarray, torch.Tensor, callable]=0,
        dtype: Optional[Union[str, torch.dtype]] = None
        ):
        DeepMemoryPolicyComponent.__init__(
            self,
            main_method_name='select_new_tokens',
            per_layer=per_layer,
            per_head=per_head,
            shared=shared,
            output_params=output_params,
            buffer_names=buffer_names,
            initializer=initializer,
            has_embedding_dim=False,
            dtype=dtype,
            )

    @abc.abstractmethod
    def select_new_tokens(
        self,
        layer_id,
        parameters,
        token_scores,
        new_sequences,
        num_new_tokens,
        attn_weights=None,
        attn_mask=None,
        position_ids=None,
        threshold_shift: float = 0.0,
        **kwargs,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Produces indexes for the selected KV cache tokens and a selection
           mask.'''
        raise NotImplementedError
    
    def get_cache_size(self,) -> Optional[int]:
        return None
    
    def register_scoring(self, scoring_network: ScoringNetwork):
        self._requires_recomputation = (
            self._requires_recomputation or
            scoring_network.requires_recomputation)
        assert scoring_network.reduced_output



