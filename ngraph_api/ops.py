#*******************************************************************************
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
#*******************************************************************************

"""Factory functions for all ngraph ops."""

import numpy as np

from typing import Optional, Set

from pyngraph import Node
from pyngraph.op import Abs, Parameter, Sqrt, Exp, Log, Negative, Floor, Ceiling, Divide, \
    Broadcast, Multiply, Subtract, Add

from ngraph_api.utils.input_validation import assert_list_of_ints
from ngraph_api.utils.types import get_element_type, py_numeric_type, TensorShape
from ngraph_api.utils import nameable_op, get_broadcast_axes


@nameable_op
def parameter(shape, dtype=np.float32, name=None):
    # type: (TensorShape, py_numeric_type, str) -> Parameter
    """Return an ngraph Parameter object."""
    assert_list_of_ints(shape, 'Parameter shape must be a list of integer values.')
    element_type = get_element_type(dtype)
    return Parameter(element_type, shape)


@nameable_op
def absolute(node, name=None):  # type: (Node, str) -> Node
    """Return node which applies f(x) = abs(x) to the input node elementwise."""
    return Abs(node)


@nameable_op
def sqrt(node, name=None):  # type: (Node, str) -> Node
    """Return node which applies square root to the input node elementwise."""
    return Sqrt(node)


@nameable_op
def exp(node, name=None):  # type: (Node, str) -> Node
    """Return node which applies exp to the input node elementwise."""
    return Exp(node)


@nameable_op
def log(node, name=None):  # type: (Node, str) -> Node
    """Return node which applies natural logarithm to the input node elementwise."""
    return Log(node)


@nameable_op
def negative(node, name=None):  # type: (Node, str) -> Node
    """Return node which applies f(x) = -x to the input node elementwise."""
    return Negative(node)


@nameable_op
def floor(node, name=None):  # type: (Node, str) -> Node
    """Return node which applies floor to the input node elementwise."""
    return Floor(node)


@nameable_op
def ceiling(node, name=None):  # type: (Node, str) -> Node
    """Return node which applies ceiling to the input node elementwise."""
    return Ceiling(node)


@nameable_op
def divide(left_node, right_node, name=None):  # type: (Node, Node, str) -> Node
    """Return node which applies f(x) = A/B to the input nodes elementwise."""
    return Divide(left_node, right_node)


@nameable_op
def multiply(left_node, right_node, name=None):  # type: (Node, Node, str) -> Node
    """Return node which applies f(x) = A*B to the input nodes elementwise."""
    return Multiply(left_node, right_node)


@nameable_op
def subtract(left_node, right_node, name=None):  # type: (Node, Node, str) -> Node
    """Return node which applies f(x) = A-B to the input nodes elementwise."""
    return Subtract(left_node, right_node)


@nameable_op
def add(left_node, right_node, name=None):  # type: (Node, Node, str) -> Node
    """Return node which applies f(x) = A+B to the input nodes elementwise."""
    return Add(left_node, right_node)


@nameable_op
def broadcast(node, new_shape, axis=None, name=None):  # type: (Node, TensorShape, int, str) -> Node
    """Return node which is broadcasts input node values to specified shape."""
    return Broadcast(node, new_shape, get_broadcast_axes(new_shape, node.shape, axis))
