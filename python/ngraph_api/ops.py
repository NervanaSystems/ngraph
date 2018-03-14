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

"""Factory functions for all ngraph ops."""

import numpy as np

from ngraph import AxisSet, AxisVector, CoordinateDiff, Node, Shape, Strides

from ngraph.op import Abs, Add, AvgPool, Broadcast, Ceiling, Constant, Convert, Convolution, \
    Divide, Dot, Equal, Exp, Floor, Greater, GreaterEq, Less, LessEq, Log, Maximum, MaxPool, \
    Minimum, Multiply, Negative, Not, NotEqual, Parameter, Reshape, Sqrt, Subtract, Sum, Tanh

from typing import Iterable, List

from ngraph_api.utils.broadcasting import get_broadcast_axes
from ngraph_api.utils.decorators import nameable_op, binary_op, unary_op
from ngraph_api.utils.input_validation import assert_list_of_ints
from ngraph_api.utils.types import NumericType, NumericData, TensorShape, make_constant_node, \
    as_node, NodeInput
from ngraph_api.utils.types import get_element_type


@nameable_op
def parameter(shape, dtype=np.float32, name=None):
    # type: (TensorShape, NumericType, str) -> Parameter
    """Return an ngraph Parameter object."""
    assert_list_of_ints(shape, 'Parameter shape must be a list of integer values.')
    element_type = get_element_type(dtype)
    return Parameter(element_type, Shape(shape))


@nameable_op
def constant(value, dtype=None, name=None):  # type: (NumericData, NumericType, str) -> Constant
    """Return an ngraph Constant object with the specified value."""
    return make_constant_node(value, dtype)


# Unary ops
@unary_op
def absolute(node, name=None):  # type: (NodeInput, str) -> Node
    """Return node which applies f(x) = abs(x) to the input node elementwise."""
    node = as_node(node)
    return Abs(node)


@unary_op
def sqrt(node, name=None):  # type: (NodeInput, str) -> Node
    """Return node which applies square root to the input node elementwise."""
    return Sqrt(node)


@unary_op
def exp(node, name=None):  # type: (NodeInput, str) -> Node
    """Return node which applies exp to the input node elementwise."""
    return Exp(node)


@unary_op
def log(node, name=None):  # type: (NodeInput, str) -> Node
    """Return node which applies natural logarithm to the input node elementwise."""
    return Log(node)


@unary_op
def negative(node, name=None):  # type: (NodeInput, str) -> Node
    """Return node which applies f(x) = -x to the input node elementwise."""
    return Negative(node)


@unary_op
def floor(node, name=None):  # type: (NodeInput, str) -> Node
    """Return node which applies floor to the input node elementwise."""
    return Floor(node)


@unary_op
def ceiling(node, name=None):  # type: (NodeInput, str) -> Node
    """Return node which applies ceiling to the input node elementwise."""
    return Ceiling(node)


@unary_op
def reshape(node, input_order, output_shape, name=None):
    # type: (Node, List[int], List[int], str) -> None
    """Return reshaped node according to provided parameters.

    :param node: The tensor we want to reshape.
    :param input_order: The order in which to iterate over input axes of input tensor.
    :param output_shape: The new shape for input tensor.
    """
    return Reshape(node, AxisVector(input_order), Shape(output_shape))


# Binary ops
@binary_op
def divide(left_node, right_node, name=None):  # type: (NodeInput, NodeInput, str) -> Node
    """Return node which applies f(x) = A/B to the input nodes elementwise."""
    return Divide(left_node, right_node)


@binary_op
def multiply(left_node, right_node, name=None):  # type: (NodeInput, NodeInput, str) -> Node
    """Return node which applies f(x) = A*B to the input nodes elementwise."""
    return Multiply(left_node, right_node)


@binary_op
def subtract(left_node, right_node, name=None):  # type: (NodeInput, NodeInput, str) -> Node
    """Return node which applies f(x) = A-B to the input nodes elementwise."""
    return Subtract(left_node, right_node)


@binary_op
def add(left_node, right_node, name=None):  # type: (NodeInput, NodeInput, str) -> Node
    """Return node which applies f(x) = A+B to the input nodes elementwise."""
    return Add(left_node, right_node)


@binary_op
def minimum(left_node, right_node, name=None):  # type: (NodeInput, NodeInput, str) -> Node
    """Return node which applies the minimum operation to input nodes elementwise."""
    return Minimum(left_node, right_node)


@binary_op
def maximum(left_node, right_node, name=None):  # type: (NodeInput, NodeInput, str) -> Node
    """Return node which applies the maximum operation to input nodes elementwise."""
    return Maximum(left_node, right_node)


# Logical ops
@binary_op
def equal(left_node, right_node, name=None):  # type: (NodeInput, NodeInput, str) -> Node
    """Return node which checks if input nodes are equal elementwise."""
    return Equal(left_node, right_node)


@binary_op
def not_equal(left_node, right_node, name=None):  # type: (NodeInput, NodeInput, str) -> Node
    """Return node which checks if input nodes are unequal elementwise."""
    return NotEqual(left_node, right_node)


@binary_op
def greater(left_node, right_node, name=None):  # type: (NodeInput, NodeInput, str) -> Node
    """Return node which checks if left input node is greater than the right node elementwise."""
    return Greater(left_node, right_node)


@binary_op
def greater_eq(left_node, right_node, name=None):  # type: (NodeInput, NodeInput, str) -> Node
    """Return node which checks if left node is greater or equal to the right node elementwise."""
    return GreaterEq(left_node, right_node)


@binary_op
def less(left_node, right_node, name=None):  # type: (NodeInput, NodeInput, str) -> Node
    """Return node which checks if left input node is less than the right node elementwise."""
    return Less(left_node, right_node)


@binary_op
def less_eq(left_node, right_node, name=None):  # type: (NodeInput, NodeInput, str) -> Node
    """Return node which checks if left node is less or equal to the right node elementwise."""
    return LessEq(left_node, right_node)


@unary_op
def logical_not(node, name=None):  # type: (Node, str) -> Node
    """Return node which applies logical negation to the input node elementwise."""
    return Not(node)


# Extend Node class to support binary operators
Node.__add__ = add
Node.__sub__ = subtract
Node.__mul__ = multiply
Node.__div__ = divide
Node.__truediv__ = divide
Node.__radd__ = lambda left, right: add(right, left)
Node.__rsub__ = lambda left, right: subtract(right, left)
Node.__rmul__ = lambda left, right: multiply(right, left)
Node.__rdiv__ = lambda left, right: divide(right, left)
Node.__rtruediv__ = lambda left, right: divide(right, left)
Node.__eq__ = equal
Node.__ne__ = not_equal
Node.__lt__ = less
Node.__le__ = less_eq
Node.__gt__ = greater
Node.__ge__ = greater_eq


# Custom ops
@nameable_op
def broadcast(node, new_shape, axis=None, name=None):  # type: (Node, TensorShape, int, str) -> Node
    """Return node which broadcasts input node values to specified shape."""
    return Broadcast(node, Shape(new_shape), get_broadcast_axes(new_shape, node.shape, axis))


@nameable_op
def convert(node, new_type, name=None):  # type: (Node, NumericType, str) -> Node
    """Return node which casts input node values to specified type."""
    new_element_type = get_element_type(new_type)
    return Convert(node, new_element_type)


# Non-linear ops
@unary_op
def tanh(node, name=None):  # type: (Node, str) -> Node
    """Return node which applies tanh to the input node elementwise."""
    return Tanh(node)


# matmul ops
@nameable_op
def dot(left_node, right_node, name=None):
    # type: (Node, Node, str) -> Node
    """Return node which performs matrix multiplication of two input nodes."""
    return Dot(left_node, right_node)


# convpool ops
@nameable_op
def convolution(x,                      # type: Node
                weights,                # type: Node
                strides=None,           # type: List[int]
                dilation=None,          # type: List[int]
                padding_above=None,     # type: List[int]
                padding_below=None,     # type: List[int]
                name=None,              # type: str
                ):
    # type: (...) -> Node
    """Return convolution node."""
    if strides is None:
        strides = [1] * (len(x.shape) - 2)  # Default to as many 1s as spatial dimensions of input.
    if dilation is None:
        dilation = [1] * (len(x.shape) - 2)
    if padding_above is None:
        padding_above = [0] * (len(x.shape) - 2)
    if padding_below is None:
        padding_below = [0] * (len(x.shape) - 2)

    return Convolution(x, weights, Strides(strides), Strides(dilation),
                       CoordinateDiff(padding_above), CoordinateDiff(padding_below))


@nameable_op
def avg_pool(x,                      # type: Node
             window_shape,           # type: TensorShape
             strides=None,           # type: List[int]
             padding_above=None,     # type: List[int]
             padding_below=None,     # type: List[int]
             zero_pad=True,          # type: bool
             name=None,              # type: str
             ):
    # type: (...) -> Node
    """Return average pooling node."""
    if strides is None:
        strides = [1] * len(window_shape)  # Default to as many 1s as spatial dimensions of input.
    if padding_above is None:
        padding_above = [0] * len(window_shape)
    if padding_below is None:
        padding_below = [0] * len(window_shape)

    return AvgPool(x, Shape(window_shape), Strides(strides),
                   Shape(padding_above), Shape(padding_above), zero_pad)


@nameable_op
def max_pool(x,                      # type: Node
             window_shape,           # type: TensorShape
             strides=None,           # type: List[int]
             padding_above=None,     # type: List[int]
             padding_below=None,     # type: List[int]
             name=None,              # type: str
             ):
    # type: (...) -> Node
    """Return max pooling node."""
    if strides is None:
        strides = [1] * len(window_shape)  # Default to as many 1s as spatial dimensions of input.
    if padding_above is None:
        padding_above = [0] * len(window_shape)
    if padding_below is None:
        padding_below = [0] * len(window_shape)

    return MaxPool(x, Shape(window_shape), Strides(strides),
                   Shape(padding_above), Shape(padding_above))


# reduction ops
@nameable_op
def sum(node, reduction_axes=None, name=None):  # type: (Node, Iterable[int], str) -> Node
    """Element-wise sums the input tensor, eliminating the specified reduction axes.

    :param reduction_axes: The axes to eliminate through summation.
    """
    if reduction_axes is None:
        reduction_axes = set(range(len(node.shape)))
    if type(reduction_axes) is not set:
        reduction_axes = set(reduction_axes)
    return Sum(node, AxisSet(reduction_axes))
