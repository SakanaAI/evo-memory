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

from utils import pad_and_concat_buffered_attn_mxs
from .shared import RegistrationCompatible, SynchronizableBufferStorage
from .base_deep_components import DeepMemoryPolicyComponent


class MemoryPolicy(
    nn.Module,
    RegistrationCompatible,
    SynchronizableBufferStorage,
    abc.ABC,
):

    def __init__(self, cache_size, init_module: bool = True):
        if init_module:
            nn.Module.__init__(self=self)
        self.cache_size = cache_size
        self.num_memory_layers = 0
        self.config = None
        self.registration_kwargs = None
        self.idxs = None
        self.cached_attn_mxs = None
        self.layer_ids = []
        self._record_eval_stats = False
        self.auxiliary_loss = None
        self.auxiliary_loss_callback = False
        self._is_in_training_mode = False
        if not hasattr(self, '_has_buffers_to_merge'):
            SynchronizableBufferStorage.__init__(
                self=self, buffers_to_merge=[], sub_buffer_storages=[])
        if not hasattr(self, 'lazy_param_num'):
            self.lazy_param_num = False

        self.initialize_buffers()

    def register_auxiliary_loss_callback(self, auxiliary_loss):
        self.auxiliary_loss = auxiliary_loss
        self.auxiliary_loss_callback = True
        print(
            'ERROR: Using auxiliary loss with memory policy with no parameters')
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f'cache_size={self.cache_size}'

    def set_params(self, params) -> None:
        pass

    @property
    def record_eval_stats(self):
        return self._record_eval_stats

    @record_eval_stats.setter
    def record_eval_stats(self, value):
        self._record_eval_stats = value

    def get_layer_params(self,):
        return None

    def is_dynamic(self,):
        return False

    def set_params_batch_idxs(self, param_idxs, ) -> None:
        pass

    def get_param_stats(self, reset=True) -> dict:
        return {}

    def log_param_stats(self,) -> dict:
        return {}

    def load_cached_attn_mxs(self, cached_attn_mxs):
        if self.requires_attn_scores:
            self.cached_attn_mxs = []
        for i, cached_attn_mx in enumerate(cached_attn_mxs):
            bs, h, n_q, n_kv = cached_attn_mx.shape
            self.update_layer_rotary_offset(
                layer_id=i, num_new_tokens=n_q, num_all_tokens=n_kv)
            if self.requires_attn_scores:
                self.cached_attn_mxs.append(cached_attn_mx)

    @property
    def n_memory_layers(self,):

        return self.num_memory_layers

    def update_rotary_offset(self, num_new_tokens, num_all_tokens,):
        for layer_id in range(self.num_memory_layers):
            self.update_layer_rotary_offset(
                layer_id=layer_id,
                num_new_tokens=num_new_tokens,
                num_all_tokens=num_all_tokens,
            )

    def update_layer_rotary_offset(self, layer_id, num_new_tokens, num_all_tokens,):

        if num_all_tokens > num_new_tokens:
            self.rotary_offset[layer_id] += num_new_tokens
        else:
            self.rotary_offset[layer_id] = num_new_tokens

    def get_rotary_offset(self, layer_id=0):
        return self.rotary_offset[layer_id].to(dtype=torch.long)

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def update_cache(self,
                     past_key_values,
                     num_new_tokens,
                     attn_weights_list,
                     attention_mask,
                     query_states=None,
                     position_ids=None,

                     analyze=False,
                     **kwargs):

        num_all_tokens = past_key_values[0][0].shape[-2]
        if not analyze:
            self.update_rotary_offset(
                num_new_tokens=num_new_tokens,
                num_all_tokens=num_all_tokens,
            )

        if analyze:
            analysis_dicts = []

        new_sequences = num_all_tokens == num_new_tokens
        use_buffered_data = not (new_sequences or self.empty_buffers)
        if use_buffered_data:
            num_new_tokens += self.buffered_tokens
            if self.requires_position_ids:
                all_pos_ids = self.buffered_pos_ids + [position_ids]
                position_ids = torch.concat(all_pos_ids, dim=-1)
        legacy_cache = ()
        for i, layer_past_key_values in enumerate(past_key_values):
            update_new_tokens = num_new_tokens
            if self.requires_attn_scores:
                layer_attn_weights = attn_weights_list[i]

                if use_buffered_data:

                    layer_attn_weights = pad_and_concat_buffered_attn_mxs(
                        buffered_attn_mxs=(self.buffered_attn_mxs[i] +
                                           [layer_attn_weights]),
                        move_to_gpu=False,
                        padding_side='right')
                if self.cached_attn_mxs is not None:
                    cached_attn = self.cached_attn_mxs[i]
                    bs, h, n_c_q, n_c_kv = cached_attn.shape
                    bs, h, n_q, n_kv = layer_attn_weights.shape
                    cached_attn = F.pad(cached_attn, pad=(0, n_kv - n_c_kv),
                                        value=0)
                    layer_attn_weights = torch.concat(
                        [cached_attn, layer_attn_weights], dim=-2)
                    update_new_tokens = num_new_tokens + n_c_q

            else:
                layer_attn_weights = None
            if self.requires_queries:
                layer_queries = query_states[i]
                if use_buffered_data:
                    buffered_queries = self.buffered_queries[i]+[layer_queries]
                    layer_queries = torch.concat(buffered_queries, dim=-2)
                kwargs['query_states'] = layer_queries

            key_cache, value_cache = layer_past_key_values

            if analyze:
                new_key_cache, new_value_cache, analysis_dict = (
                    self.analyze(
                        layer_id=i,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        num_new_tokens=update_new_tokens,
                        attn_weights=layer_attn_weights,
                        attn_mask=attention_mask,
                        position_ids=position_ids,
                        **kwargs)
                )
                analysis_dicts.append(analysis_dict)
            else:
                new_key_cache, new_value_cache = self.update_layer_cache(
                    layer_id=i,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    num_new_tokens=update_new_tokens,
                    attn_weights=layer_attn_weights,
                    attn_mask=attention_mask,
                    position_ids=position_ids,
                    **kwargs)

            legacy_cache += ((new_key_cache, new_value_cache),)

        if self.auxiliary_loss_callback:
            self.auxiliary_loss.memory_policy_update_callback(
                pop_idxs=self._flat_param_idxs,
                new_sequences=new_sequences,
                new_kv_cache=legacy_cache,)

        if not analyze:
            self.cached_attn_mxs = None
            self.initialize_buffers()

        if analyze:
            return legacy_cache, analysis_dicts
        else:
            return legacy_cache

    def analyze(self, layer_id, key_cache, value_cache,
                num_new_tokens, attn_weights, attn_mask=None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        analysis_dict = {}
        new_key_cache, new_value_cache = self.update_layer_cache(
            layer_id=layer_id,
            key_cache=key_cache,
            value_cache=value_cache,
            num_new_tokens=num_new_tokens,
            attn_weights=attn_weights,
            attn_mask=attn_mask,
            **kwargs,
        )
        return new_key_cache, new_value_cache, analysis_dict

    def initialize_buffers(self,):
        self.buffered_tokens = 0
        if self.requires_attn_scores:
            self.buffered_attn_mxs = [[]
                                      for _ in range(self.num_memory_layers)]
        if self.requires_position_ids:
            self.buffered_pos_ids = []
        if self.requires_queries:
            self.buffered_queries = [[] for _ in range(self.num_memory_layers)]
        self.empty_buffers = True

    def update_buffers(
            self,
            num_new_tokens,
            attn_weights_list,
            query_states,
            position_ids):
        if self.requires_attn_scores:
            for i, attn_weights in enumerate(attn_weights_list):
                self.buffered_attn_mxs[i].append(attn_weights)
        if self.requires_position_ids:
            self.buffered_pos_ids.append(position_ids)
        if self.requires_queries:
            for i, queries in enumerate(query_states):
                self.buffered_queries[i].append(queries)
        self.empty_buffers = False
        self.buffered_tokens += num_new_tokens

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def buffer_cache(self,
                     past_key_values,
                     num_new_tokens,
                     attn_weights_list,
                     attention_mask,
                     query_states=None,
                     position_ids=None,
                     **kwargs):

        num_all_tokens = past_key_values[0][0].shape[-2]
        new_sequences = num_all_tokens == num_new_tokens
        if new_sequences:
            self.initialize_buffers()
        self.update_buffers(
            num_new_tokens=num_new_tokens,
            attn_weights_list=attn_weights_list,
            query_states=query_states,
            position_ids=position_ids,
        )
        self.update_rotary_offset(
            num_new_tokens=num_new_tokens,
            num_all_tokens=num_all_tokens,
        )

    def process_attn_weights(self, attn_weights):
        '''If GQA group different attention heads and average attention score'''
        if self.num_key_value_groups is not None:
            bs, heads, n_q, n_kv = attn_weights.shape
            attn_weights = attn_weights.view(
                bs, self.num_heads, self.num_key_value_groups, n_q, n_kv)
            attn_weights = attn_weights.mean(2)
        return attn_weights

    @abc.abstractmethod
    def update_layer_cache(self, layer_id, key_cache, value_cache,
                           num_new_tokens, attn_weights, attn_mask=None,
                           **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        '''key_cache and value_cache are the full updated current states,
           new_keys and new_values are (redundant) tensors with only the mha
           keys and values (short-term memory)'''
        raise NotImplementedError

    def finalize_registration(self,):

        assert self.config is not None
        assert self.registration_kwargs is not None
        self.register_buffer('rotary_offset',
                             tensor=torch.zeros([self.num_memory_layers],
                                                dtype=torch.long),
                             )
        self.param_size = 0
        if self.requires_attn_scores:

            self.cumulated_attn_tensors = [None for _ in range(
                self.num_memory_layers)]

        if self.requires_position_ids:
            self.cache_position_ids = [None for _ in range(
                self.num_memory_layers)]

        if self.is_diversity_based:
            self.stored_div_tensors = [None for _ in range(
                self.num_memory_layers)]

    def get_init_param_values(self,):
        return None

    @property
    def requires_attn_scores(self,):
        return False

    @property
    def requires_queries(self,):

        return False

    @property
    def requires_position_ids(self,):
        return False

    @property
    def is_diversity_based(self,):
        return False


class Recency(MemoryPolicy):
    def __init__(self, cache_size):
        super().__init__(cache_size=cache_size)
        if cache_size == None:
            self.limit_cache = False
        else:
            self.limit_cache = True

    def update_layer_cache(self, layer_id, key_cache, value_cache, num_new_tokens,
                           attn_weights, attn_mask=None, **kwargs,) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.limit_cache:
            return (key_cache[..., -self.cache_size:, :],
                    value_cache[..., -self.cache_size:, :])
        else:
            return key_cache, value_cache

    @property
    def requires_attn_scores(self,):

        return False


class AttnRequiringRecency(MemoryPolicy):
    '''Isolating effect of flash attn vs. regular attn implementations'''

    def __init__(self, cache_size):
        super().__init__(cache_size=cache_size)

    def update_layer_cache(self, layer_id, key_cache, value_cache, num_new_tokens,
                           attn_weights, attn_mask=None, **kwargs,) -> Tuple[torch.Tensor, torch.Tensor]:
        return (key_cache[..., -self.cache_size:, :],
                value_cache[..., -self.cache_size:, :])

    @property
    def requires_attn_scores(self,):
        return True


class ParamMemoryPolicy(MemoryPolicy):
    def __init__(
            self,
            cache_size, base_param_size, pop_size,
            per_head, per_layer, additional_shared_params=0,
            learnable_params: Optional[Dict[str, Union[str, tuple]]] = None,
            learned_params: Optional[Dict[str, bool]] = None,
            component_names: Optional[List[str]] = None,
            init_module: bool = True,
            lazy_param_num: bool = False,
    ):
        MemoryPolicy.__init__(
            self,
            cache_size=cache_size,
            init_module=init_module,
        )
        self.per_head = per_head
        self.per_layer = per_layer
        if per_head:
            assert per_layer, ('does not make sense to have parameters per head'
                               ' and not per layer')
        self.pop_size = pop_size
        self.auxiliary_loss = None
        self.auxiliary_loss_callback = False

        self.learned_params = learned_params
        self.component_names = component_names

        self.learnable_params = learnable_params
        self.shared_param_counter = 0
        self.nonshared_param_counter = 0

        self.initial_base_params = []
        self.initial_additional_params = []
        self.learnable_params_getter_names = dict()
        self.components_params_getter_names = dict()
        self.lazy_param_num = lazy_param_num

        if learnable_params is not None or component_names is not None:
            if base_param_size > 0 or additional_shared_params > 0:
                print("WARNING: specificifying learnable/learned params will " +
                      "override base param size/additional shared params")
            self.additional_shared_params = additional_shared_params
            self.learnable_params = learnable_params
            self.learned_params = learned_params
            assert (isinstance(self.learned_params, dict) or
                    isinstance(self.component_names, list))
            self.setup_params()
        else:
            self.base_param_size = base_param_size
            self.additional_shared_params = additional_shared_params
            self.has_done_automatic_setup = False
            self.components_have_been_setup = False

    def register_auxiliary_loss_callback(self, auxiliary_loss):
        self.auxiliary_loss = auxiliary_loss
        self.auxiliary_loss_callback = True

    def setup_params(self,):
        if self.learned_params is not None:
            learned_param_keys = set(self.learned_params.keys())
            learnable_param_keys = set(self.learnable_params.keys())

            for k in learned_param_keys:
                assert k in learnable_param_keys, (
                    f'ERROR - specified learned {k} not in set of learnable ' +
                    f' params: {learnable_param_keys}')
            assert len(learnable_param_keys) > 0

            for param_name, param_scaling in self.learnable_params.items():
                if hasattr(self, param_name):
                    initial_value = getattr(self, param_name)
                else:
                    initial_value = getattr(self, 'initial_' + param_name)
                if param_name in self.learned_params:
                    self.setup_learned_param(
                        param_name=param_name,
                        initial_value=initial_value,
                        shared=self.learned_params[param_name],
                        scaling=param_scaling,
                    )
                else:
                    self.setup_nonlearned_param(
                        param_name=param_name,
                        value=initial_value,
                    )
        if self.component_names is not None:
            self.setup_components()
        self.has_done_automatic_setup = True
        self.base_param_size = len(self.initial_base_params)
        self.additional_shared_params = len(self.initial_additional_params)

        print('New number of initial base params from setup: '
              f'{self.base_param_size}')
        print('New number of initial additional params from setup: '
              f'{self.additional_shared_params}')

    def setup_learned_param(self, param_name, initial_value, shared,
                            scaling):
        curr_shared_counter = copy.copy(self.shared_param_counter)
        curr_nonshared_counter = copy.copy(self.nonshared_param_counter)
        exp_scaling = False
        bounded = False
        bounded_min = 0
        bounded_range = 1
        print(f'Initializing learned {param_name}')
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

            norm_initial_value = (initial_value - bounded_min)/bounded_range
            initial_value = np.log(norm_initial_value) - np.log(
                1 - norm_initial_value)
        if shared:
            self.initial_additional_params.append(initial_value)
            self.shared_param_counter += 1
        else:
            self.initial_base_params.append(initial_value)
            self.nonshared_param_counter += 1

        def getter(params, additional_params):
            if shared:

                v = additional_params[..., [curr_shared_counter]]
            else:

                v = params[..., [curr_nonshared_counter]]

            if exp_scaling:
                v = torch.exp(v)
            elif bounded:
                v = torch.sigmoid(v)*bounded_range + bounded_min
            return v

        getter_name = 'get_' + param_name
        setattr(self, getter_name, getter)
        self.learnable_params_getter_names[param_name] = getter_name

    def setup_nonlearned_param(self, param_name, value):
        setattr(self, param_name, value)

        def getter(params, additional_params):
            return getattr(self, param_name)
        getter_name = 'get_' + param_name
        setattr(self, getter_name, getter)
        self.learnable_params_getter_names[param_name] = getter_name

    def setup_components(self,):
        for component_name in self.component_names:
            self.setup_learned_component(component_name=component_name)

        self.base_param_size = len(self.initial_base_params)
        self.additional_shared_params = len(self.initial_additional_params)

        print('New number of initial base params from components setup: '
              f'{self.base_param_size}')
        print('New number of initial additional params from components setup: '
              f'{self.additional_shared_params}')

        self.components_have_been_setup = True

    def setup_learned_component(
            self,
            component_name: str,
            shared: Optional[bool] = None,
            initial_values: Optional[Union[torch.Tensor, np.ndarray]] = None,
            scaling: Optional[str] = None,
    ):

        if hasattr(self, component_name):
            component: DeepMemoryPolicyComponent = getattr(
                self, component_name)
        else:
            print(f'ERROR: {component_name} not stored into the state ' +
                  'dictionary of the memory policy')
            raise NotADirectoryError

        if shared is None:
            shared = component.shared
        if initial_values is None:
            initial_values = component.get_param_initial()
        if scaling is None:
            scaling = component.get_param_scaling()

        if isinstance(initial_values, torch.Tensor):
            initial_values = initial_values.cpu().numpy()

        base_param_dims = component.get_param_size()

        assert initial_values.shape[-1] == base_param_dims

        curr_shared_counter = copy.copy(self.shared_param_counter)
        curr_nonshared_counter = copy.copy(self.nonshared_param_counter)
        exp_scaling = False
        bounded = False
        bounded_min = 0
        bounded_range = 1
        if scaling == None:
            scaling = 'linear'
        print(f'Initializing learned weights for {component_name}')
        if isinstance(scaling, str):
            assert scaling in ['linear', 'exp', 'composite']
            if scaling == 'exp':
                exp_scaling = True
                assert np.all(initial_values > 1e-7)
                initial_values = np.log(initial_values)
            if scaling == 'composite':
                num_linear, num_exp, num_bounded, (
                    min_bound, max_bound) = component.get_param_scaling_dims()
                has_linear = num_linear > 0
                has_exp = num_exp > 0
                has_bounded = num_bounded > 0

                raise NotImplementedError
        else:
            min_max_list = list(scaling)
            assert len(min_max_list) == 2
            assert np.all(min_max_list[0] < initial_values)
            assert np.all(min_max_list[1] > initial_values)
            bounded_min = min_max_list[0]
            bounded_range = min_max_list[1] - min_max_list[0]
            bounded = True

            norm_initial_value = (initial_values - bounded_min)/bounded_range
            initial_values = np.log(norm_initial_value) - np.log(
                1 - norm_initial_value)

        shared_end_idx = curr_shared_counter + base_param_dims
        nonshared_end_idx = curr_nonshared_counter + base_param_dims

        def getter(params, additional_params):
            if shared:
                v = additional_params[..., curr_shared_counter:shared_end_idx]
            else:
                v = params[..., curr_nonshared_counter:nonshared_end_idx]
            if exp_scaling:
                v = torch.exp(v)
            elif bounded:
                v = torch.sigmoid(v)
            return v

        if shared:
            self.initial_additional_params += initial_values.tolist()
            self.shared_param_counter = shared_end_idx
        else:
            self.initial_base_params += initial_values.tolist()
            self.nonshared_param_counter = nonshared_end_idx
        getter_name = 'get_params_for_' + component_name
        setattr(self, getter_name, getter)
        self.components_params_getter_names[component_name] = getter_name

    def register_model_to_components(self,):
        '''Pass on registration values to component, and setup buffers and other
           misc. after registration'''
        for component_name in self.component_names:
            print(f'Registering {component_name}')
            print(self.token_embedding)
            component: DeepMemoryPolicyComponent = getattr(
                self, component_name)
            component.register_new_memory_model(
                config=self.config,
                registration_kwargs=self.registration_kwargs)
            component.finalize_registration()

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
            component: DeepMemoryPolicyComponent = getattr(
                self, component_name)
            component_stats = component.get_param_stats(
                parameters=parameters, reset=reset)
            for k, v in component_stats.items():
                stats[component_stats_prefix + k] = v
        return stats

    def get_init_param_values_post_setup(self,):
        if not self.has_done_automatic_setup:
            raise NotImplementedError

        initial_base_params = torch.tensor(self.initial_base_params)
        initial_additional_params = self.initial_additional_params
        initial_base_params = initial_base_params.view(
            1, 1, self.base_param_size).expand(
                self.param_layer_dim, self.param_head_dim, -1)

        initial_base_params = torch.reshape(initial_base_params,
                                            shape=[self.total_base_param_size,])
        initial_additional_params = torch.tensor(initial_additional_params)

        return torch.concat([initial_base_params, initial_additional_params],
                            dim=0)

    def get_layer_learnable_params_dict_post_setup(self, layer_id):
        gathered_layer_params = self.get_layer_params(layer_id=layer_id)

        gathered_shared_params = self.get_additional_shared_params()

        gathered_shared_params = gathered_shared_params.unsqueeze(-2)
        params_dict = {}
        for p_name, p_getter in self.learnable_params_getter_names.items():
            params_dict[p_name] = getattr(self, p_getter)(
                params=gathered_layer_params,
                additional_params=gathered_shared_params)

        return params_dict

    def set_params(self, params) -> None:
        if self.lazy_param_num and self.param_pop_size is None:
            lazy_pop_dim = params.shape[0]
            self.initialize_pop_params_buffers(param_pop_size=lazy_pop_dim)
            self.to(device=params.device)
        params, pop_shared_params = params.split(
            [self.total_base_param_size, self.additional_shared_params], dim=-1)
        reshaped_params = params.view(self.param_pop_size,
                                      self.param_layer_dim,
                                      self.param_head_dim,
                                      self.base_param_size)
        self.pop_params.data.copy_(reshaped_params)
        self.pop_shared_params.data.copy_(pop_shared_params)

    def get_layer_params(self, layer_id):

        if self.per_layer:
            layer_params = self.pop_params[:, layer_id]
        else:
            layer_params = self.pop_params[:, 0]

        gathered_params = torch.gather(layer_params, dim=0,
                                       index=self.base_param_idxs)

        return gathered_params

    def get_additional_shared_params(self,):
        gathered_params = torch.gather(self.pop_shared_params, dim=0,
                                       index=self.shared_param_idxs)

        return gathered_params

    def set_params_batch_idxs(self, param_idxs) -> None:
        if not isinstance(param_idxs, torch.Tensor):
            param_idxs = torch.tensor(param_idxs)
        param_idxs = param_idxs.to(device=self.pop_params.device,
                                   dtype=torch.long)
        self._flat_param_idxs = param_idxs.view(-1)
        base_param_idxs = param_idxs.view(-1, 1, 1).expand(
            -1, self.param_head_dim, self.base_param_size)
        self.base_param_idxs = base_param_idxs

        shared_param_idxs = param_idxs.view(-1, 1).expand(
            -1, self.additional_shared_params)
        self.shared_param_idxs = shared_param_idxs

    def initialize_pop_params_buffers(self, param_pop_size):
        self.param_pop_size = param_pop_size
        self.register_buffer('pop_params',
                             tensor=torch.zeros([self.param_pop_size,
                                                 self.param_layer_dim,
                                                 self.param_head_dim,
                                                 self.base_param_size,]))
        self.register_buffer('pop_shared_params',
                             tensor=torch.zeros([self.param_pop_size,
                                                 self.additional_shared_params,]))

    def finalize_registration(self,):

        MemoryPolicy.finalize_registration(self,)
        if self.per_layer:
            self.param_layer_dim = self.num_memory_layers
        else:
            self.param_layer_dim = 1
        if self.per_head:
            assert 'num_heads' in self.registration_kwargs, (
                'need to pass the number of heads during registration')
            self.param_head_dim = self.registration_kwargs['num_heads']
        else:
            self.param_head_dim = 1

        if self.lazy_param_num:
            self.param_pop_size = None
        else:
            self.initialize_pop_params_buffers(param_pop_size=self.pop_size)
        self.total_base_param_size = (
            self.param_layer_dim * self.param_head_dim * self.base_param_size)
        self.param_size = self.additional_shared_params + self.total_base_param_size
        self.base_param_idxs = None
        self._flat_param_idxs = None

    def get_param_value_stats_post_setup(self,) -> dict:
        stats = dict()
        shared_param_dict = dict()
        fixed_param_dict = dict()
        shared_stats_key_prefix = f'mem_stats/params/shared/'
        fixed_stats_key_prefix = f'mem_stats/params/fixed/'
        for i in range(self.param_layer_dim):
            layer_param_dict = self.get_layer_learnable_params_dict_post_setup(
                layer_id=i)

            if self.per_layer:
                stats_key_prefix = f'mem_stats/params/layer_id_{i}/'
            else:
                stats_key_prefix = shared_stats_key_prefix

            not_shared_param_dict = dict()
            for param_name, value in layer_param_dict.items():
                if param_name in self.learned_params:
                    is_shared = self.learned_params[param_name]
                    if is_shared:
                        shared_param_dict[param_name] = value
                    else:
                        not_shared_param_dict[param_name] = value
                else:
                    fixed_param_dict[param_name] = value

            for k, v in not_shared_param_dict.items():
                stats[stats_key_prefix + k + '_mean'] = v.mean().item()
                if self.per_head:
                    stats[stats_key_prefix + k + '_head_std'] = v.std(-2).mean(
                    ).item()

        for k, v in shared_param_dict.items():
            stats[shared_stats_key_prefix + k + '_mean'] = v.mean().item()

        for k, v in fixed_param_dict.items():
            value = None
            if isinstance(v, torch.Tensor):
                value = v.mean().item()
            elif isinstance(v, list) or isinstance(v, np.ndarray):
                try:
                    value = np.mean(v)
                except:
                    pass
            elif isinstance(v, int) or isinstance(v, float):
                value = v

            if value is not None:
                stats[fixed_stats_key_prefix + k + '_mean'] = value

        return stats
