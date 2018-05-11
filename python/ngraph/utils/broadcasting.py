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
import logging
from typing import List

import ngraph as ng

from ngraph.impl import Node, AxisSet
from ngraph.utils.types import TensorShape, get_dtype, make_constant_node, NodeInput

log = logging.getLogger(__file__)


def get_broadcast_axes(output_shape, input_shape, axis=None):
    # type: (TensorShape, TensorShape, int) -> AxisSet
    """Generate a list of broadcast axes for ngraph++ broadcast.

    Informally, a broadcast "adds" axes to the input tensor,
    replicating elements from the input tensor as needed to fill the new dimensions.
    Function calculate which of the output axes are added in this way.

    :param output_shape: The new shape for the output tensor.
    :param input_shape: The shape of input tensor.
    :param axis: The axis along which we want to replicate elements.
    :return: The indices of added axes.
    """
    axes_indexes = list(range(0, len(output_shape)))
    if axis is None:
        output_begin = len(output_shape) - len(input_shape)
    else:
        output_begin = axis
    right_axes_indexes = list(range(output_begin, output_begin + len(input_shape)))
    for index in reversed(right_axes_indexes):
        del axes_indexes[index]
    return AxisSet(set(axes_indexes))


def as_elementwise_compatible_nodes(*input_values):  # type: (*NodeInput) -> List[Node]
    """Return all input values as ngraph Nodes with the same shape and element type.

    Scalar values will be converted to ngraph Constant Nodes.
    """
    input_nodes = [node for node in input_values
                   if issubclass(type(node), Node)]  # type: List[Node]

    if not input_nodes:
        raise NotImplementedError('Operations on scalars only are not supported.')

    shapes = {tuple(node.shape) for node in input_nodes}
    if len(shapes) > 1:
        log.warning('More than one different shape in input nodes %s.', input_nodes)

    types = [node.get_element_type() for node in input_nodes]
    unique_types = {repr(type) for type in types}
    if len(unique_types) > 1:
        log.warning('More than one different data type in input nodes %s.', input_nodes)

    sorted_shapes = sorted(shapes, key=len)
    broadcast_shape = sorted_shapes.pop()
    broadcast_dtype = get_dtype(types.pop())

    output_nodes = []
    for input_value in input_values:
        if issubclass(type(input_value), Node):
            input_value = ng.broadcast(input_value, broadcast_shape)
            output_nodes.append(input_value)
        else:
            input_value = make_constant_node(input_value, dtype=broadcast_dtype)
            output_nodes.append(ng.broadcast(input_value, broadcast_shape))

    return output_nodes
