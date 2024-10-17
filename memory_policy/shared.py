from typing import Optional, Tuple, Union, Dict, List

from collections import OrderedDict
import abc
import copy
import torch
from torch import nn


class RegistrationCompatible(abc.ABC):
    def register_auxiliary_loss_callback(self, auxiliary_loss):
        self.auxiliary_loss = auxiliary_loss
        self.auxiliary_loss_callback = True
        print(
            'ERROR: Using auxiliary loss with component with no parameters')
        raise NotImplementedError

    def register_new_memory_layer(self, config, registration_kwargs):
        
        
        
        
        curr_layer_id = self.num_memory_layers
        self.num_memory_layers = self.num_memory_layers + 1
        self.config = config
        self.registration_kwargs = registration_kwargs
        return curr_layer_id

    def register_new_memory_model(self, config, registration_kwargs):
        assert 'num_memory_layers' in registration_kwargs
        self.num_memory_layers = registration_kwargs['num_memory_layers']
        self.num_heads = registration_kwargs['num_heads']
        self.num_key_value_groups = registration_kwargs.get(
            'num_key_value_groups', None)
        self.config = config
        self.registration_kwargs = registration_kwargs
        self.hidden_size = registration_kwargs.get(
            'hidden_size', None)
        if self.hidden_size is None:
            if hasattr(config, 'hidden_size'):
                self.hidden_size = config.hidden_size
            else:
                raise NotImplementedError


class SynchronizableBufferStorage(abc.ABC):
    def __init__(
            self,
            buffers_to_merge: Union[List[str], dict] = [],
            sub_buffer_storages: list = [],
    ):
        self.initialize_buffer_dicts_to_merge(
            buffers_to_merge=buffers_to_merge,
            sub_buffer_storages=sub_buffer_storages)
        self.training_mode()
        self.unfreeze_sync_buffers()

    def are_sync_buffers_frozen(self,):
        return self._frozen_sync_buffers

    def freeze_sync_buffers(self, freeze=True):
        self._frozen_sync_buffers = freeze
        if self._has_sub_buffers_to_merge:
            for sub_buffer_n, sub_buffer in (
                    self._sub_buffer_ordered_references.items()):
                sub_buffer.freeze_sync_buffers(freeze=freeze)

    def unfreeze_sync_buffers(self,):
        self.freeze_sync_buffers(freeze=False)

    def training_mode(self,):
        self._is_in_training_mode = True
        if self._has_sub_buffers_to_merge:
            for sub_buffer_n, sub_buffer in (
                    self._sub_buffer_ordered_references.items()):
                sub_buffer.training_mode()

    def evaluation_mode(self,):
        self._is_in_training_mode = False
        if self._has_sub_buffers_to_merge:
            for sub_buffer_n, sub_buffer in (
                    self._sub_buffer_ordered_references.items()):
                sub_buffer.evaluation_mode()

    def get_buffers_to_merge_keys(self,):
        return self._buffers_to_merge_keys

    
    def get_buffers_list(self,):
        buffers_dict = self.get_buffers_dict()
        assert len(buffers_dict) == len(self._buffers_to_merge_keys)
        return [buffers_dict[k] for k in self._buffers_to_merge_keys]

    def merge_buffers_list(
            self,
            buffers_to_merge: List[List[torch.Tensor]],
    ) -> List[torch.Tensor]:
        merged_buffers = []
        buffer_group_idx = 0
        if self._has_owned_buffers_to_merge:
            merged_buffers += self._merge_own_buffers(
                buffers_to_merge=buffers_to_merge[
                    :self._num_total_owned_buffers_to_merge])
            buffer_group_idx += self._num_total_buffers_to_merge
        if self._has_sub_buffers_to_merge:
            for n_buffers, (sub_buffer_n, sub_buffer) in zip(
                    self.num_buffers_per_sub_buffer,
                    self._sub_buffer_ordered_references.items()):

                end_buffer_group_idx = buffer_group_idx + n_buffers
                rel_buffers_to_merge = buffers_to_merge[
                    buffer_group_idx:end_buffer_group_idx]
                merged_buffers += sub_buffer.merge_buffers_list(
                    buffers_to_merge=rel_buffers_to_merge)
                buffer_group_idx = end_buffer_group_idx
        assert len(merged_buffers) == len(buffers_to_merge)
        return merged_buffers

    def _merge_own_buffers(
            self,
            buffers_to_merge: List[List[torch.Tensor]],
    ) -> List[torch.Tensor]:
        raise NotImplementedError

    def receive_buffers_list(self, buffers_list):
        assert len(buffers_list) == len(self._buffers_to_merge_keys)
        buffers_dict = {k: v for k, v in zip(
            self._buffers_to_merge_keys, buffers_list)}
        self.load_buffers_dict(buffers_dict=buffers_dict)

    def get_buffers_dict(self,):
        buffers_dict = {}
        if self._has_owned_buffers_to_merge:
            buffers_dict.update(self.buffers_to_merge_dict)
        if self._has_sub_buffers_to_merge:
            buffers_dict.update(self.get_dict_from_sub_buffers())
        return buffers_dict

    def load_buffers_dict(self, buffers_dict):
        if len(buffers_dict) == 0:
            return
        else:
            assert set(buffers_dict.keys()) == set(
                self.get_buffers_to_merge_keys())

            for k in self.buffers_to_merge_dict:
                self.buffers_to_merge_dict[k] = buffers_dict[k]
            if self._has_sub_buffers_to_merge:
                self.load_dict_to_sub_buffers(buffers_dict=buffers_dict)

    def _self_merge_own_buffers(self,) -> List[torch.Tensor]:
        raise NotImplementedError

    def self_merge(self,) -> List[torch.Tensor]:
        merged_buffers = []
        if self._has_owned_buffers_to_merge:
            merged_buffers += self._self_merge_own_buffers()
        if self._has_sub_buffers_to_merge:
            for k, sub_buffer in self._sub_buffer_ordered_references.items():
                merged_buffers += sub_buffer.self_merge()
        return merged_buffers

    def initialize_buffer_dicts_to_merge(
            self, buffers_to_merge: Union[List[str], dict],
            sub_buffer_storages: list,
            reset: bool = True,
    ):

        assert reset
        if isinstance(buffers_to_merge, dict):
            self.buffers_to_merge_dict = OrderedDict(buffers_to_merge)
        else:
            self.buffers_to_merge_dict = OrderedDict(
                [(k, 0) for k in buffers_to_merge])

        self._buffers_to_merge_keys = list(self.buffers_to_merge_dict.keys())
        self._owned_buffers_to_merge_keys = copy.copy(
            self._buffers_to_merge_keys)
        self._num_total_owned_buffers_to_merge = len(
            self.buffers_to_merge_dict)
        self._num_total_buffers_to_merge = (
            self._num_total_owned_buffers_to_merge)
        self._has_buffers_to_merge = self._has_owned_buffers_to_merge = (
            self._num_total_buffers_to_merge > 0)
        self.register_sub_buffers_to_merge(
            sub_buffer_storages=sub_buffer_storages,)

    def register_sub_buffers_to_merge(self, sub_buffer_storages: list):
        self.num_buffers_per_sub_buffer = []
        self._buffers_to_merge_keys_from_sub_buffers = []
        self._sub_buffer_ordered_references: Dict[
            str, SynchronizableBufferStorage] = OrderedDict()
        for i, buffer in enumerate(sub_buffer_storages):
            if isinstance(buffer, str):
                assert hasattr(self, buffer)
                buffer_obj: SynchronizableBufferStorage = getattr(self, buffer)
                buffer_name = buffer
            else:
                buffer_obj: SynchronizableBufferStorage = buffer
                buffer_name = f'{i}'

            if buffer_obj._has_buffers_to_merge:
                sub_buffer_keys = buffer_obj._buffers_to_merge_keys
                self._buffers_to_merge_keys_from_sub_buffers += [
                    buffer_name + '_' + k for k in sub_buffer_keys]
                self.num_buffers_per_sub_buffer.append(len(sub_buffer_keys))
                self._sub_buffer_ordered_references[buffer_name] = buffer_obj

        self._buffers_to_merge_keys += (
            self._buffers_to_merge_keys_from_sub_buffers)

        self._num_total_buffers_to_merge_from_sub_buffers = sum(
            self.num_buffers_per_sub_buffer)
        self._has_sub_buffers_to_merge = (
            self._num_total_buffers_to_merge_from_sub_buffers > 0)
        self._has_buffers_to_merge = (
            self._has_buffers_to_merge or self._has_sub_buffers_to_merge)
        self._num_total_buffers_to_merge += (
            self._num_total_buffers_to_merge_from_sub_buffers)

    def get_dict_from_sub_buffers(self,) -> Dict[str, torch.Tensor]:
        buffers_dict = OrderedDict()
        for buffer_name, buffer_obj in (
                self._sub_buffer_ordered_references.items()):
            for k, v in buffer_obj.get_buffers_dict().items():
                buffers_dict[buffer_name + '_' + k] = v
        return buffers_dict

    def load_dict_to_sub_buffers(self, buffers_dict) -> Dict[str, torch.Tensor]:
        for buffer_name, buffer_obj in (
                self._sub_buffer_ordered_references.items()):

            buffer_sub_dict = {}
            for k, v in buffers_dict.items():
                target_prefix = f'{buffer_name}_'
                if k.startswith(target_prefix):
                    buffer_sub_dict[k.removeprefix(target_prefix)] = v

            buffer_obj.load_buffers_dict(buffer_sub_dict)

        return buffers_dict
