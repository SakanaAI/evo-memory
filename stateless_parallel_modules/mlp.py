import os
import pdb
import copy
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, Callable


import torch
from torch import nn
import torch.utils.checkpoint

from .base import (
    StatelessGeneralizedOperation, GeneralizedLinear,
    StatelessGeneralizedModule)

from utils import get_nonlinearity


class StatelessGeneralizedMLP(StatelessGeneralizedModule):
    def __init__(
            self,
            input_features: Optional[int],
            hidden_features: Optional[int],
            output_features: Optional[int],
            # depth = 1, means a single linear operation
            hidden_depth: int,
            bias: bool,
            # base_parallel_operations: Optional[int] = None,
            non_linearity: Union[str, Callable] = 'relu',
            # add a residual connection for the hidden layers
            residual: bool = True,
            # add a residual connection from the first layer
            residual_first: bool = False,
    ):

        StatelessGeneralizedModule.__init__(
            self=self,
            input_features=input_features,
            output_features=output_features,
            init_module=True,
        )

        self.hidden_features = hidden_features
        self.hidden_depth = hidden_depth
        self.bias = bias

        self.non_linearity = get_nonlinearity(
            nonlinearity=non_linearity,
        )

        self.residual_first = residual_first
        self.residual = residual

        self.is_linear = hidden_depth == 1
        self.has_intermediate = hidden_depth > 2

        if (self.input_features is not None and
            self.output_features is not None and
                self.hidden_features is not None):
            print('instantiating ops')
            self.instantiate_and_setup_ops(
                input_features=input_features,
                hidden_features=hidden_features,
                output_features=output_features,
                preceding_module=None,
                default_output_features_mult=1,
            )

    def instantiate_and_setup_ops(
            self,
            input_features: Optional[int] = None,
            hidden_features: Optional[int] = None,
            output_features: Optional[int] = None,
            preceding_module=None,
            default_output_features_mult: int = 1,
            **kwargs,
    ):

        if (self.input_features is None or
            self.output_features is None or
                self.hidden_features is None):

            self.instantiate_model(
                input_features=input_features,
                output_features=output_features,
                preceding_module=preceding_module,
                default_output_features_mult=default_output_features_mult,
            )
            if self.hidden_features is None and hidden_features is not None:
                self.hidden_features = hidden_features
            elif self.hidden_features is None:
                print('Warning: hidden features not specified setting to ' +
                      f'{self.output_features} (output features)')
                self.hidden_features = self.output_features

        if self.residual_first:
            # assert hidden_depth > 1
            assert self.input_features == self.hidden_features

        self.linear_op = GeneralizedLinear()

        in_dims = self.input_features
        self.input_output_features_tuple = []
        for _ in range(self.hidden_depth - 1):
            out_dims = self.hidden_features
            self.input_output_features_tuple.append(
                (in_dims, out_dims))
            in_dims = self.hidden_features
        out_dims = self.output_features
        self.input_output_features_tuple.append((in_dims, out_dims))

        operation_list = [
            self.linear_op for _ in range(self.hidden_depth)]
        operation_kwargs = dict(
            bias=self.bias,
        )
        operation_kwargs_overrides_list = [
            dict(in_features=in_dims, out_features=out_dims)
            for in_dims, out_dims in self.input_output_features_tuple]

        self.setup_operations(
            operations=operation_list,
            operation_kwargs=operation_kwargs,
            operation_kwargs_overrides_list=operation_kwargs_overrides_list,
            save_as_module_list=True,
        )

    def forward(
            self,
            # parallel_operations x batch_size x input_features
            inputs: torch.Tensor,
            *args,
            n_parallel_dimensions: Optional[int] = None,
            **kwargs,
    ):

        if n_parallel_dimensions is not None:
            input_shape = inputs.shape
            inputs = inputs.flatten(
                start_dim=0, end_dim=n_parallel_dimensions-1)
            # this is guaranteed to be 3-dimensional
            inputs = inputs.flatten(start_dim=1, end_dim=-2)

        weight, bias = self.parameters_per_layer[0]
        # raise NotImplementedError
        h = inputs
        # TODO
        h_out = self.linear_op(
            input=h,
            weight=weight,
            bias=bias,
            parallel_operations=self.parallel_operations,
            # initial input already flattened above
            n_parallel_dimensions=None,
        )

        if not self.is_linear:
            h_out = self.non_linearity(h_out)
            if self.residual_first:
                h = h + h_out
            else:
                h = h_out
            if self.has_intermediate:
                for weight, bias in self.parameters_per_layer[1:-1]:
                    h_out = self.linear_op(
                        input=h,
                        weight=weight,
                        bias=bias,
                        parallel_operations=self.parallel_operations,
                        # initial input already flattened above
                        n_parallel_dimensions=None,
                    )
                    h_out = self.non_linearity(h)
                    if self.residual:
                        h = h + h_out
                    else:
                        h = h_out
            weight, bias = self.parameters_per_layer[-1]
            h_out = self.linear_op(
                input=h,
                weight=weight,
                bias=bias,
                parallel_operations=self.parallel_operations,
                # initial input already flattened above
                n_parallel_dimensions=None,
            )
        if n_parallel_dimensions is not None:
            h_out = h_out.view(*input_shape[:-1], self.output_features)
        return h_out
