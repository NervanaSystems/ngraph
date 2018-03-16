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
from typing import Optional, List

import ngraph as ng

from ngraph_bind import AxisSet, Node
from ngraph.utils.types import TensorShape, get_dtype, make_constant_node, NodeInput

log = logging.getLogger(__file__)


def get_broadcast_axes(left_shape, right_shape, axis):
    # type: (TensorShape, TensorShape, Optional[int]) -> AxisSet
    """Generate a list of broadcast axes for ngraph++ broadcast.

    Informally, a broadcast "adds" axes to the input tensor,
    replicating elements from the input tensor as needed to fill the new dimensions.
    Function calculate which of the output axes is being so added.
    For example, an output shape of `{2,5,6,2,8}` and input shape of `{2,6}` means
    that the broadcast axes must be `{1,3,4}`.
    """
    axes_indexes = list(range(0, len(left_shape)))
    if axis is None:
        right_begin = len(left_shape) - len(right_shape)
    else:
        right_begin = axis
    right_axes_indexes = list(range(right_begin, right_begin + len(right_shape)))
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

    types = {node.get_element_type() for node in input_nodes}
    if len(types) > 1:
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
