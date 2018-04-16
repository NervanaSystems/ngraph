# ******************************************************************************
# Copyright 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
"""Provide a layer of abstraction for the ngraph++ runtime environment."""
import logging
from typing import List

import numpy as np

from ngraph.impl import Function, Node, serialize, TensorViewType, util
from ngraph.impl.runtime import Backend
from ngraph.impl.op import Parameter

from ngraph.utils.types import get_dtype, NumericData

log = logging.getLogger(__file__)


def runtime(backend_name='CPU'):  # type: (str) -> 'Runtime'
    """Create a Runtime object (helper factory).

    Use signature to parameterize runtime as needed.
    """
    return Runtime(backend_name)


class Runtime:
    """Represents the ngraph++ runtime environment."""

    def __init__(self, backend_name):  # type: (str) -> None
        self.backend_name = backend_name
        self.backend = Backend.create(backend_name)

    def __repr__(self):  # type: () -> str
        return '<Runtime: Backend=\'{}\'>'.format(self.backend_name)

    def computation(self, node, *inputs):  # type: (Node, *Node) -> 'Computation'
        """Return a callable Computation object."""
        return Computation(self, node, *inputs)


class Computation:
    """ngraph callable computation object."""

    def __init__(self, runtime, node, *parameters):  # type: (Runtime, Node, *Parameter) -> None
        self.runtime = runtime
        self.node = node
        self.parameters = parameters
        self.tensor_views = []  # type: List[TensorViewType]
        for parameter in parameters:
            shape = parameter.get_shape()
            element_type = parameter.get_element_type()
            self.tensor_views.append(runtime.backend.create_tensor(element_type, shape))
        self.function = Function(self.node, self.parameters, 'ngraph_computation')
        self.backend = runtime.backend

    def __repr__(self):  # type: () -> str
        params_string = ', '.join([param.name for param in self.parameters])
        return '<Computation: {}({})>'.format(self.node.name, params_string)

    def __call__(self, *input_values):  # type: (*NumericData) -> NumericData
        """Run computation on input values and return result."""
        for tensor_view, value in zip(self.tensor_views, input_values):
            if not isinstance(value, np.ndarray):
                value = np.array(value)
            Computation._write_ndarray_to_tensor_view(value, tensor_view)

        result_element_type = self.node.get_element_type()
        result_shape = self.node.get_shape()
        result_dtype = get_dtype(result_element_type)

        result_view = self.runtime.backend.create_tensor(
            result_element_type, result_shape)
        result_arr = np.empty(result_shape, dtype=result_dtype)

        self.backend.call(self.function, [result_view], self.tensor_views)

        Computation._read_tensor_view_to_ndarray(result_view, result_arr)
        result_arr = result_arr.reshape(result_shape)
        return result_arr

    def serialize(self, indent=0):  # type: (int) -> str
        """Serialize function (compute graph) to a JSON string.

        :param indent: set indent of serialized output
        :return: serialized model
        """
        return serialize(self.function, indent)

    @staticmethod
    def _get_buffer_size(element_type, element_count):  # type: (TensorViewType, int) -> int
        return int((element_type.bitwidth / 8.0) * element_count)

    @staticmethod
    def _write_ndarray_to_tensor_view(value, tensor_view):
        # type: (np.ndarray, TensorViewType) -> None
        tensor_view_dtype = get_dtype(tensor_view.element_type)
        if value.dtype != tensor_view_dtype:
            log.warning(
                'Attempting to write a %s value to a %s tensor. Will attempt type conversion.',
                value.dtype,
                tensor_view.element_type)
            value = value.astype(tensor_view_dtype)

        buffer_size = Computation._get_buffer_size(
            tensor_view.element_type, tensor_view.element_count)
        tensor_view.write(util.numpy_to_c(np.ascontiguousarray(value)), 0, buffer_size)

    @staticmethod
    def _read_tensor_view_to_ndarray(tensor_view, output):
        # type: (TensorViewType, np.ndarray) -> None
        buffer_size = Computation._get_buffer_size(
            tensor_view.element_type, tensor_view.element_count)
        tensor_view.read(util.numpy_to_c(output), 0, buffer_size)
