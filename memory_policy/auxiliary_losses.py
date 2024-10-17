import os
import pdb
import copy
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

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
from .base_dynamic import DynamicMemoryPolicy, DynamicParamMemoryPolicy

from omegaconf import OmegaConf, DictConfig
import hydra
import numpy as np


class MemoryPolicyAuxiliaryLoss(nn.Module, abc.ABC):

    def __init__(self,
                 memory_policy: ParamMemoryPolicy,
                 coeff: float = 1.0,
                 adaptive_target: Optional[float] = None,
                 adaptive_target_param: str = 'exp',
                 optimizer=None,
                 adaptive_update_every: int = 1,
                 device: str = 'cuda',
                 ):
        nn.Module.__init__(self,)
        self.device = device
        assert isinstance(memory_policy, ParamMemoryPolicy)
        self.pop_size = memory_policy.param_pop_size

        self.num_memory_layers = memory_policy.num_memory_layers

        self.step_losses = torch.zeros([self.pop_size], dtype=torch.long,
                                       device=device)

        self.is_adaptive = adaptive_target is not None
        self.adaptive_target = adaptive_target

        assert adaptive_target_param in ['exp', 'linear']
        self.is_exp = adaptive_target_param == 'exp'
        self.adaptive_target_param = adaptive_target_param
        if self.is_adaptive:
            assert optimizer is not None
            if self.is_exp:
                assert coeff > 0
                init_coeff = np.log(coeff)
            else:
                init_coeff = coeff
            self.raw_coeff = nn.Parameter(
                data=torch.zeros([1]) + float(init_coeff), requires_grad=False)
        else:
            self.register_buffer('raw_coeff', tensor=torch.tensor(coeff),
                                 persistent=False)
        self.adaptive_update_every = adaptive_update_every
        self.num_adaptive_updates = 0
        self.stored_losses = []

        self.num_samples_per_pop = torch.zeros(
            [self.pop_size], dtype=torch.long, device=device)
        self.total_losses_per_pop = torch.zeros(
            [self.pop_size], dtype=torch.long, device=device)
        memory_policy.register_auxiliary_loss_callback(auxiliary_loss=self)

    def restart_recording(self,):

        self.num_samples_per_pop.data.copy_(torch.zeros_like(
            self.num_samples_per_pop))
        self.total_losses_per_pop.data.copy_(torch.zeros_like(
            self.total_losses_per_pop))

    @abc.abstractmethod
    def memory_policy_layer_callback(
            self,
            layer_id,
            pop_idxs,
            new_sequences,
            key_cache,
            value_cache,
            dynamic_mask=None):
        raise NotImplementedError

    @abc.abstractmethod
    def memory_policy_update_callback(
            self,
            layer_id,
            pop_idxs,
            new_sequences,
            new_kv_cache,):
        raise NotImplementedError

    @property
    def coeff(self,):
        if self.is_adaptive and self.is_exp:
            return torch.exp(self.raw_coeff)
        else:
            return self.raw_coeff

    @abc.abstractmethod
    def get_loss(self,):
        raise NotImplementedError

    def optim_params(self, loss):
        self.optimizer.zero_grad(set_to_none=True)
        dual_difference = self.adaptive_target - loss
        self.raw_coeff.grad = dual_difference
        self.optimizer.step()
        return dual_difference

    def setup_optimizer(self, optimizer):
        assert optimizer is not None
        if isinstance(self.optimizer, DictConfig):

            self.optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
                optimizer, params=[self.raw_coeff], _convert_='all')
        else:
            self.optimizer: torch.optim.Optimizer = optimizer(
                params=[self.raw_coeff])

    def forward(self,):
        loss = self.get_loss()

        if self.is_adaptive:
            if self.num_adaptive_updates % self.adaptive_update_every == 0:
                _ = self.optim_params(loss=loss)
            self.num_adaptive_updates += 1
        scaled_loss = self.coeff*loss

        return scaled_loss


class JointAuxiliaryLosses(MemoryPolicyAuxiliaryLoss):

    def __init__(self, memory_policy: ParamMemoryPolicy,
                 auxiliary_losses: List[MemoryPolicyAuxiliaryLoss]):
        MemoryPolicyAuxiliaryLoss.__init__(self, memory_policy=memory_policy)
        assert len(auxiliary_losses) > 0
        self.auxiliary_losses = nn.ModuleList(auxiliary_losses)
        self.adaptive_losses = [
            loss for loss in auxiliary_losses if loss.is_adaptive]
        self.is_adaptive = len(self.adaptive_losses) > 0

    def get_loss(self,):
        loss = 0
        for aux_loss in self.auxiliary_losses:
            loss = loss + aux_loss.get_loss()
        return loss


class SparsityAuxiliaryLoss(MemoryPolicyAuxiliaryLoss):

    def __init__(self,
                 memory_policy: DynamicParamMemoryPolicy,
                 coeff: float = 1.0,
                 adaptive_target: Optional[float] = None,
                 adaptive_target_param: str = 'exp',
                 optimizer=None,
                 adaptive_update_every: int = 1,
                 sparsity_mode: str = 'mean',
                 sparsity_per_head: bool = False,
                 device: str = 'cuda',
                 ):
        MemoryPolicyAuxiliaryLoss.__init__(
            self,
            memory_policy=memory_policy,
            coeff=coeff,
            adaptive_target=adaptive_target,
            adaptive_target_param=adaptive_target_param,
            optimizer=optimizer,
            adaptive_update_every=adaptive_update_every,
            device=device,
        )
        self.sparsity_mode = sparsity_mode
        assert sparsity_mode in ['mean',]
        self.sparsity_per_head = sparsity_per_head

        self.losses_per_pop_per_layer = torch.zeros(
            [self.num_memory_layers, self.pop_size], dtype=torch.long,
            device=device)

    def memory_policy_layer_callback(
            self,
            layer_id,
            pop_idxs,
            new_sequences,
            key_cache,
            value_cache,
            dynamic_mask=None,
            scoring_network_params=None):

        if dynamic_mask is not None:
            unmasked_samples_per_head: torch.Tensor = dynamic_mask.to(
                dtype=self.losses_per_pop_per_layer.dtype).sum(-1)
            if self.sparsity_per_head:

                layer_loss = (unmasked_samples_per_head.sum(-1) /
                              unmasked_samples_per_head.numel()).to(
                                  dtype=torch.long)
            else:
                layer_loss = unmasked_samples_per_head.max(-1)[0]
        else:
            layer_loss = key_cache.size(-2)*torch.ones(
                [key_cache.size(0)],
                device=self.losses_per_pop_per_layer.device,
                dtype=self.losses_per_pop_per_layer.dtype,
            )
        self.losses_per_pop_per_layer[layer_id].data.copy_(torch.zeros_like(
            self.losses_per_pop_per_layer[layer_id]))
        self.losses_per_pop_per_layer[layer_id].scatter_add_(
            dim=0, index=pop_idxs, src=layer_loss,)  # reduce='sum', include_self=False)

    def memory_policy_update_callback(
            self,
            pop_idxs,
            new_sequences,
            new_kv_cache,):
        losses_per_pop = self.losses_per_pop_per_layer.sum(dim=0)

        self.num_samples_per_pop.scatter_reduce_(
            dim=0, index=pop_idxs,
            # .to(dtype=self.num_samples_per_), #[pop_idxs], device=self.num_samples_per_pop.device),
            src=torch.ones_like(pop_idxs).to(dtype=losses_per_pop.dtype),
            reduce='sum',
            include_self=True)
        self.total_losses_per_pop.data.add_(losses_per_pop)

    def get_loss(self,):
        return self.total_losses_per_pop/(
            torch.clamp_min_(
                self.num_samples_per_pop*self.num_memory_layers, 1))


class L2NormAuxiliaryLoss(MemoryPolicyAuxiliaryLoss):
    def __init__(self,
                 memory_policy: DynamicParamMemoryPolicy,
                 coeff: float = 1.0,
                 adaptive_target: Optional[float] = None,
                 adaptive_target_param: str = 'linear',
                 optimizer=None,
                 adaptive_update_every: int = 1,
                 device: str = 'cuda',
                 ):
        MemoryPolicyAuxiliaryLoss.__init__(
            self,
            memory_policy=memory_policy,
            coeff=coeff,
            adaptive_target=adaptive_target,
            adaptive_target_param=adaptive_target_param,
            optimizer=optimizer,
            adaptive_update_every=adaptive_update_every,
            device=device,
        )

        self.losses_per_pop_per_layer = torch.zeros(
            [self.num_memory_layers, self.pop_size], device=device)

        self.total_losses_per_pop = torch.zeros(
            [self.pop_size], dtype=torch.float, device=device)

    def memory_policy_layer_callback(
            self,
            layer_id,
            pop_idxs,
            new_sequences,
            key_cache,
            value_cache,
            dynamic_mask=None,
            scoring_network_params=None):

        self.total_losses_per_pop[pop_idxs] = (
            scoring_network_params ** 2).mean().to(self.device)

    def memory_policy_update_callback(
            self,
            pop_idxs,
            new_sequences,
            new_kv_cache,):
        pass

    def get_loss(self,):
        return self.total_losses_per_pop
