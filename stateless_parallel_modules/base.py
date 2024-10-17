import os
import pdb
import copy
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, Callable, List
import abc

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import LlamaPreTrainedModel
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from utils import get_nonlinearity


class StatelessGeneralizedOperation(nn.Module, abc.ABC):
    def __init__(
            self,
    ):
        nn.Module.__init__(self=self)

    @abc.abstractmethod
    def total_parameters(
        self,
        *args,
        parallel_operations: Optional[int] = None,
        **kwargs,
    ) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def prepare_parameters(
            parameters: torch.Tensor,
            *args,
            parallel_operations: Optional[int] = None,
            **kwargs,
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def forward(
            # parallel_operations x batch_size x in_features
            input: torch.Tensor,
            *args,
            parallel_operations: Optional[int] = None,
            n_parallel_dimensions: Optional[int] = None,
            **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError


class GeneralizedLinear(StatelessGeneralizedOperation):
    def __init__(
            self,
    ):
        StatelessGeneralizedOperation.__init__(self=self,)

    def total_parameters(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        parallel_operations: Optional[int] = None,
        **kwargs,
    ) -> int:
        total_weight_dimension = in_features*out_features
        total_base_parameters_dimension = total_weight_dimension
        if bias:
            total_base_parameters_dimension += out_features
        if parallel_operations is not None:
            total_parameters_dimension = (
                parallel_operations*total_base_parameters_dimension)
        else:
            total_parameters_dimension = total_base_parameters_dimension
        return total_parameters_dimension, total_base_parameters_dimension

    def prepare_parameters(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        parameters: torch.Tensor,
        parallel_operations: Optional[int] = None,
        **kwargs,
    ):
        if parallel_operations is not None:
            parameters = parameters.view(parallel_operations, -1)
            if bias:
                total_weight_dimension = in_features*out_features
                flat_w, flat_b = parameters.split_with_sizes(
                    [total_weight_dimension, out_features], dim=-1)
                # parallel_ops x 1 x out_features
                b = flat_b.unsqueeze_(-2)
            else:
                flat_w = parameters
                b = None
            # shape required for baddbmm/addbmm
            w = flat_w.view(parallel_operations, in_features, out_features)
        else:
            if bias:
                total_weight_dimension = in_features*out_features
                flat_w, b = parameters.split_with_sizes(
                    [total_weight_dimension, out_features], dim=-1)
            else:
                flat_w = parameters
                b = None
            # shape required for F.linear
            w = flat_w.view(out_features, in_features)
        return w, b

    def forward(
        self,
        input: torch.Tensor,  # parallel_operations x batch_size x in_features
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        parallel_operations: Optional[int] = None,
        n_parallel_dimensions: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        if n_parallel_dimensions is not None:
            input_shape = input.shape

            # this is guaranteed to be 3-dimensional
            input = input.flatten(start_dim=1, end_dim=-2)

        if parallel_operations is not None:
            if bias is not None:
                out = torch.baddbmm(input=bias, batch1=input, batch2=weight)
            else:
                out = input @ weight
        else:
            out = F.linear(input=input, weight=weight, bias=bias)
        if n_parallel_dimensions is not None:
            out = out.view(*input_shape[:-1], -1)
        return out


class StatelessGeneralizedModule(nn.Module, abc.ABC):
    def __init__(
            self,
            input_features: Optional[int],
            output_features: Optional[int],
            init_module: bool = True,
            operation_kwargs: Optional[dict] = None
    ):
        if init_module:
            nn.Module.__init__(self=self,)

        if operation_kwargs is None:
            self.operation_kwargs = {}
        else:
            self.operation_kwargs = operation_kwargs

        self.input_features = input_features
        self.output_features = output_features
        if not hasattr(self, 'total_base_parameter_dims'):
            # num. of params needed for forward (for each parallel op.)
            self.total_base_parameter_dims = 0

        if not hasattr(self, 'n_base_parameters_per_layer'):
            self.n_base_parameters_per_layer: List[int] = []
        if not hasattr(self, 'parameters_per_layer'):
            self.parameters_per_layer: List[torch.Tensor] = []
        if not hasattr(self, 'kwargs_per_op'):
            self.kwargs_per_op: List[dict] = []
        if not hasattr(self, 'generalized_ops'):
            self.generalized_ops: List[StatelessGeneralizedOperation] = []
        self.parallel_operations: Optional[int] = None

    def get_buffer_names(self,):
        return []

    def instantiate_and_setup_ops(
        self,
        input_features: Optional[int] = None,
        output_features: Optional[int] = None,
        preceding_module=None,
        default_output_features_mult: int = 1,
        **kwargs,
    ):
        raise NotImplementedError

    def setup_operations(
            self,
            operations: Union[
                List[StatelessGeneralizedOperation],
                StatelessGeneralizedOperation],
            operation_kwargs: Optional[dict] = None,
            operation_kwargs_overrides_list: Optional[List[dict]] = None,
            # save list of operations as nn.ModuleList
            save_as_module_list: bool = True,
    ):

        if operation_kwargs is not None:
            self.operation_kwargs = operation_kwargs

        operation_kwargs = self.operation_kwargs

        if operation_kwargs_overrides_list is not None:
            assert len(operation_kwargs_overrides_list) == len(operations)
        elif isinstance(operations, StatelessGeneralizedModule):
            operation_kwargs_overrides_list = [{}]
        else:
            operation_kwargs_overrides_list = [{} for _ in operations]

        if isinstance(operations, StatelessGeneralizedModule):
            operations = [operations for _ in range(
                operation_kwargs_overrides_list)]

        self.n_base_parameters_per_layer = self.n_base_parameters_per_layer
        self.generalized_ops = operations
        for op, op_kwargs_overrides in zip(
                self.generalized_ops, operation_kwargs_overrides_list):
            op: StatelessGeneralizedOperation
            op_kwargs = {}
            op_kwargs.update(operation_kwargs)
            op_kwargs.update(op_kwargs_overrides)
            self.kwargs_per_op.append(op_kwargs)
            op_base_params, _ = op.total_parameters(
                parallel_operations=None, **op_kwargs)
            self.n_base_parameters_per_layer.append(op_base_params)

        self.total_base_parameter_dims = int(sum(
            self.n_base_parameters_per_layer))
        if save_as_module_list:
            self.generalized_ops_module_list = nn.ModuleList(
                self.generalized_ops)

    def instantiate_model(
            self,
            input_features: Optional[int] = None,
            output_features: Optional[int] = None,
            preceding_module=None,
            default_output_features_mult: int = 1,
    ):
        if self.input_features is None and input_features is not None:
            self.input_features = input_features
        elif self.input_features is None:
            assert preceding_module is not None
            assert hasattr(preceding_module, 'output_features')
            self.input_features: int = preceding_module.output_features
            print('Warning: input features not specified setting to ' +
                  f'{self.input_features} (output features of ' +
                  'preceding module)')

        assert isinstance(self.input_features, int)
        if self.output_features is None and output_features is not None:
            self.output_features = output_features
        elif self.output_features is None:
            self.output_features = (
                self.input_features*default_output_features_mult)
            print('Warning: output features not specified setting to ' +
                  f'{self.output_features} ' +
                  '(input features*default_output_features_mult)')

    def format_parameters(
            self,
            parameters: torch.Tensor,
            # should match first dimension of parameters
            parallel_operations: Optional[int] = None,
    ):

        if parallel_operations is not None:
            # sanity check, unneded/redundant (trivial overhead)
            parameters = parameters.view(
                parallel_operations, self.n_base_parameters_per_layer)

        self.parallel_operations = parallel_operations

        # assumes parameters is a flattened tensor
        flat_parameters_per_layer = parameters.split_with_sizes(
            self.n_base_parameters_per_layer, dim=-1)

        parameters_per_layer: list[torch.Tensor] = []

        for op, op_kwargs, layer_params in zip(
            self.generalized_ops,
            self.kwargs_per_op,
            flat_parameters_per_layer,
        ):

            prepared_params = op.prepare_parameters(
                parameters=layer_params,
                parallel_operations=parallel_operations,
                **op_kwargs,
            )

            parameters_per_layer.append(prepared_params)
        return parameters_per_layer

    def load_parameters(
            self,
            parameters: torch.Tensor,
            # should match first dimension of parameters
            parallel_operations: Optional[int] = None,
    ):

        if parallel_operations is not None:
            # sanity check, unneded/redundant (trivial overhead)
            parameters = parameters.view(
                parallel_operations, self.total_base_parameter_dims)

        self.parallel_operations = parallel_operations

        # assumes parameters is a flattened tensor
        flat_parameters_per_layer = parameters.split_with_sizes(
            self.n_base_parameters_per_layer, dim=-1)

        self.parameters_per_layer: list[torch.Tensor] = []

        for op, op_kwargs, layer_params in zip(
            self.generalized_ops,
            self.kwargs_per_op,
            flat_parameters_per_layer,
        ):

            prepared_params = op.prepare_parameters(
                **op_kwargs,
                parameters=layer_params,
                parallel_operations=parallel_operations,
            )

            self.parameters_per_layer.append(prepared_params)

    @abc.abstractmethod
    def forward(
        self,
        # parallel_operations x batch_size x input_features
        inputs: torch.Tensor,
        *args,
        n_parallel_dimensions: Optional[int] = None,
        **kwargs,
    ):
        raise NotImplementedError
