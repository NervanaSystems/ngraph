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

from ngraph.impl import AxisSet, AxisVector, Coordinate, CoordinateDiff, Node, NodeVector, \
    Shape, Strides

from ngraph.impl.op import Abs, Acos, Add, Asin, Atan, AvgPool, Broadcast, Ceiling, Concat, \
    Constant, Convert, Convolution, Cos, Cosh, Divide, Dot, Equal, Exp, Floor, Greater, GreaterEq, \
    Less, LessEq, Log, Max, Maximum, MaxPool, Min, Minimum, Multiply, Negative, Not, NotEqual, \
    Parameter, Product, Reshape, Slice, Softmax, Sqrt, Subtract, Sum, Tanh

from typing import Iterable, List

from ngraph.utils.broadcasting import get_broadcast_axes
from ngraph.utils.decorators import nameable_op, binary_op, unary_op
from ngraph.utils.input_validation import assert_list_of_ints
from ngraph.utils.reduction import get_reduction_axes
from ngraph.utils.types import NumericType, NumericData, TensorShape, make_constant_node, \
    NodeInput
from ngraph.utils.types import get_element_type


@nameable_op
def parameter(shape, dtype=np.float32, name=None):
    # type: (TensorShape, NumericType, str) -> Parameter
    """Return an ngraph Parameter object."""
    assert_list_of_ints(shape, 'Parameter shape must be a list of integer values.')
    element_type = get_element_type(dtype)
    return Parameter(element_type, Shape(shape))


@nameable_op
def constant(value, dtype=None, name=None):  # type: (NumericData, NumericType, str) -> Constant
    """Create a Constant node from provided value.

    :param value: One of: array of values or scalar to initialize node with.
    :param dtype: The data type of provided data.
    :param name: Optional name for output node.
    :return: The Constant node initialized with provided data.
    """
    return make_constant_node(value, dtype)


# Unary ops
@unary_op
def absolute(node, name=None):  # type: (NodeInput, str) -> Node
    """Return node which applies f(x) = abs(x) to the input node element-wise.

    :param node: One of: input node, array or scalar.
    :param name: Optional new name for output node.
    :return: New node with Abs operation applied on it.
    """
    return Abs(node)


@unary_op
def acos(node, name=None):  # type: (NodeInput, str) -> Node
    """Apply inverse cosine function on the input node element-wise.

    :param node: One of: input node, array or scalar.
    :param name: Optional new name for output node.
    :return: New node with arccos operation applied on it.
    """
    return Acos(node)


@unary_op
def asin(node, name=None):  # type: (NodeInput, str) -> Node
    """Apply inverse sine function on the input node element-wise.

    :param node: One of: input node, array or scalar.
    :param name: Optional new name for output node.
    :return: New node with arcsin operation applied on it.
    """
    return Asin(node)


@unary_op
def atan(node, name=None):  # type: (NodeInput, str) -> Node
    """Apply inverse tangent function on the input node element-wise.

    :param node: One of: input node, array or scalar.
    :param name: Optional new name for output node.
    :return: New node with arctan operation applied on it.
    """
    return Atan(node)


@unary_op
def cos(node, name=None):  # type: (NodeInput, str) -> Node
    """Apply cosine function on the input node element-wise.

    :param node: One of: input node, array or scalar.
    :param name: Optional new name for output node.
    :return: New node with cos operation applied on it.
    """
    return Cos(node)


@unary_op
def cosh(node, name=None):  # type: (NodeInput, str) -> Node
    """Apply hyperbolic cosine function on the input node element-wise.

    :param node: One of: input node, array or scalar.
    :param name: Optional new name for output node.
    :return: New node with cosh operation applied on it.
    """
    return Cosh(node)


@unary_op
def sqrt(node, name=None):  # type: (NodeInput, str) -> Node
    """Return node which applies square root to the input node elementwise."""
    return Sqrt(node)


@unary_op
def exp(node, name=None):  # type: (NodeInput, str) -> Node
    """Return node which applies exp to the input node element-wise.

    :param node: The node providing data for operation.
    :param name: The optional name for new output node.
    :return: The new node performing natural exponential operation.
    """
    return Exp(node)


@unary_op
def log(node, name=None):  # type: (NodeInput, str) -> Node
    """Return node which applies natural logarithm to the input node element-wise.

    :param node: The input node providing data for operation.
    :param name: The optional new name for output node.
    :return: The new node performing log operation element-wise.
    """
    return Log(node)


@unary_op
def negative(node, name=None):  # type: (NodeInput, str) -> Node
    """Return node which applies f(x) = -x to the input node elementwise."""
    return Negative(node)


@unary_op
def floor(node, name=None):  # type: (NodeInput, str) -> Node
    """Return node which applies floor to the input node element-wise.

    :param node: The input node providing data.
    :param name: The optional name for new output node.
    :return: The node performing element-wise floor operation.
    """
    return Floor(node)


@unary_op
def ceiling(node, name=None):  # type: (NodeInput, str) -> Node
    """Return node which applies ceiling to the input node element-wise.

    :param node: The node providing data to ceiling operation.
    :param name: Optional name for output node.
    :return: The node performing element-wise ceiling.
    """
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
    """Return node which applies f(x) = A/B to the input nodes element-wise.

    :param left_node: The node providing dividend data.
    :param right_node: The node providing divisor data.
    :param name: Optional name for output node.
    :return: The node performing element-wise division.
    """
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
    """Return node which applies f(x) = A+B to the input nodes element-wise."""
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
    """Return node which checks if input nodes are equal element-wise.

    :param left_node: The first input node for equal operation.
    :param right_node: The second input node for equal operation.
    :param name: The optional name for output new node.
    :return: The node performing element-wise equality check.
    """
    return Equal(left_node, right_node)


@binary_op
def not_equal(left_node, right_node, name=None):  # type: (NodeInput, NodeInput, str) -> Node
    """Return node which checks if input nodes are unequal elementwise."""
    return NotEqual(left_node, right_node)


@binary_op
def greater(left_node, right_node, name=None):  # type: (NodeInput, NodeInput, str) -> Node
    """Return node which checks if left input node is greater than the right node element-wise.

    :param left_node: The first input node providing data.
    :param right_node: The second input node providing data.
    :param name: The optional new name for output node.
    :return: The node performing element-wise check whether left_node is greater than right_node.
    """
    return Greater(left_node, right_node)


@binary_op
def greater_eq(left_node, right_node, name=None):  # type: (NodeInput, NodeInput, str) -> Node
    """Return node which checks if left node is greater or equal to the right node element-wise.

    :param left_node: The first input node providing data.
    :param right_node: The second input node providing data.
    :param name: The optional new name for output node.
    :return: The node performing element-wise check whether left_node is greater than or equal
             right_node.
    """
    return GreaterEq(left_node, right_node)


@binary_op
def less(left_node, right_node, name=None):  # type: (NodeInput, NodeInput, str) -> Node
    """Return node which checks if left input node is less than the right node element-wise.

    :param left_node: The first input node providing data.
    :param right_node: The second input node providing data.
    :param name: The optional new name for output node.
    :return: The node performing element-wise check whether left_node is less than the right_node.
    """
    return Less(left_node, right_node)


@binary_op
def less_eq(left_node, right_node, name=None):  # type: (NodeInput, NodeInput, str) -> Node
    """Return node which checks if left input node is less or equal the right node element-wise.

    :param left_node: The first input node providing data.
    :param right_node: The second input node providing data.
    :param name: The optional new name for output node.
    :return: The node performing element-wise check whether left_node is less than or equal the
             right_node.
    """
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
    """Return node which broadcasts input node values to specified shape.

    :param node: The node with input tensor data.
    :param new_shape: The new shape we want to broadcast tensor to.
    :param axis: The axis along which we perform broadcasting.
    :param name: Optional new name for output node.
    :return: New node with broadcasted shape.
    """
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
def dot(left_node, right_node, reduction_axes_count=None, name=None):
    # type: (Node, Node, int, str) -> Node
    """Return node which performs generalized dot product of two input nodes.

    This operation is capable of performing scalar-tensor, matrix-vector product and matrix
    multiplication.

    :param left_node: The node providing left hand side data.
    :param right_node: The node providing right hand side data.
    :param reduction_axes_count: The number of axes to reduce during dot-product.
    :param name: The optional name for output node.
    :return: The new node performing dot-product on input two nodes.
    """
    if reduction_axes_count is None:
        return Dot(left_node, right_node)
    else:
        return Dot(left_node, right_node, reduction_axes_count)


# convpool ops
@nameable_op
def convolution(data_batch,                     # type: Node
                filter_weights,                 # type: Node
                filter_strides=None,            # type: List[int]
                filter_dilation_strides=None,   # type: List[int]
                padding_below=None,             # type: List[int]
                padding_above=None,             # type: List[int]
                data_dilation_strides=None,     # type: List[int]
                name=None,                      # type: str
                ):
    # type: (...) -> Node
    """Return node performing batched convolution operation.

    :param data_batch: The node providing data batch tensor.
    :param filter_weights: The node providing filters tensor.
    :param filter_strides: The kernel window movement strides.
    :param filter_dilation_strides: The filters dilation strides.
    :param padding_below: The number of zero padding elements to add on each axis below 0
                          coordinate.
    :param padding_above: The number of zero padding elements to add on each axis above max
                          coordinate.
    :param data_dilation_strides: The data batch dilation strides.
    :param name: The optional new name for output node.
    :return: New node performing batched convolution operation.
    """
    spatial_dim_count = len(data_batch.shape) - 2
    if filter_strides is None:
        filter_strides = [1] * spatial_dim_count
    if filter_dilation_strides is None:
        filter_dilation_strides = [1] * spatial_dim_count
    if padding_above is None:
        padding_above = [0] * spatial_dim_count
    if padding_below is None:
        padding_below = [0] * spatial_dim_count
    if data_dilation_strides is None:
        data_dilation_strides = [1] * spatial_dim_count

    return Convolution(data_batch, filter_weights, Strides(filter_strides),
                       Strides(filter_dilation_strides), CoordinateDiff(padding_below),
                       CoordinateDiff(padding_above), Strides(data_dilation_strides))


@nameable_op
def avg_pool(data_batch,             # type: Node
             window_shape,           # type: TensorShape
             window_strides=None,    # type: List[int]
             padding_below=None,     # type: TensorShape
             padding_above=None,     # type: TensorShape
             include_padding=False,  # type: bool
             name=None,              # type: str
             ):
    # type: (...) -> Node
    """Return average pooling node.

    :param data_batch: The input node providing data.
    :param window_shape: The pooling window shape.
    :param window_strides: The window movement strides.
    :param padding_below: The input data optional padding below filled with zeros.
    :param padding_above: The input data optional padding below filled with zeros.
    :param include_padding: Whether or not to include zero padding in average computations.
    :param name: Optional name for the new output node.
    :return: New node with AvgPool operation applied on its data.
    """
    spatial_dim_count = len(window_shape)
    if window_strides is None:
        window_strides = [1] * spatial_dim_count
    if padding_above is None:
        padding_above = [0] * spatial_dim_count
    if padding_below is None:
        padding_below = [0] * spatial_dim_count

    return AvgPool(data_batch, Shape(window_shape), Strides(window_strides), Shape(padding_below),
                   Shape(padding_above), include_padding)


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
                   Shape(padding_above), Shape(padding_below))


# reduction ops
@nameable_op
def sum(node, reduction_axes=None, name=None):
    # type: (Node, Iterable[int], str) -> Node
    """Element-wise sums the input tensor, eliminating the specified reduction axes.

    :param reduction_axes: The axes to eliminate through summation.
    """
    reduction_axes = get_reduction_axes(node, reduction_axes)
    return Sum(node, AxisSet(reduction_axes))


@nameable_op
def max(node, reduction_axes=None, name=None):
    # type: (Node, Iterable[int], str) -> Node
    """Max-reduction operation on input tensor, eliminating the specified reduction axes.

    :param node: The tensor we want to max-reduce.
    :param reduction_axes: The axes to eliminate through max operation.
    :param name: Optional name for output node.
    """
    reduction_axes = get_reduction_axes(node, reduction_axes)
    return Max(node, AxisSet(reduction_axes))


@nameable_op
def min(node, reduction_axes=None, name=None):
    # type: (Node, Iterable[int], str) -> Node
    """Min-reduction operation on input tensor, eliminating the specified reduction axes.

    :param node: The tensor we want to max-reduce.
    :param reduction_axes: The axes to eliminate through min operation.
    :param name: Optional name for output node.
    """
    reduction_axes = get_reduction_axes(node, reduction_axes)
    return Min(node, AxisSet(reduction_axes))


@nameable_op
def prod(node, reduction_axes=None, name=None):
    # type: (Node, Iterable[int], str) -> Node
    """Product-reduction operation on input tensor, eliminating the specified reduction axes.

    :param node: The tensor we want to product-reduce.
    :param reduction_axes: The axes to eliminate through product operation.
    :param name: Optional name for output node.
    """
    reduction_axes = get_reduction_axes(node, reduction_axes)
    return Product(node, AxisSet(reduction_axes))


# reshape ops
@nameable_op
def slice(node, lower_bounds, upper_bounds, strides=None, name=None):
    # type: (Node, List[int], List[int], List[int], str) -> Node
    """Take a slice of an input tensor, (sub-tensor) that resides within a bounding box.

    Optionally this function may be provided with stride along each axis.

    :param node: The tensor we want to slice.
    :param lower_bounds: The (inclusive) lower-bound coordinates for the tensor slice.
    :param upper_bounds: The (exclusive) upper-bound coordinates for the tensor slice.
    :param strides: The strides for the tensor slice.
    :param name: Optional name for the output node.
    :return: Return node that represents a slice of input nodes data.
    """
    if strides is None:
        return Slice(node, Coordinate(lower_bounds), Coordinate(upper_bounds))
    else:
        return Slice(node, Coordinate(lower_bounds), Coordinate(upper_bounds), Strides(strides))


@nameable_op
def concat(nodes, axis, name=None):  # type: (List[Node], int, str) -> Node
    """Concatenate input nodes into single new node along specified axis.

    :param nodes: The nodes we want concatenate into single new node.
    :param axis: The axis along which we want to concatenate input nodes.
    :param name: The optional new name for output node.
    :return: Return new node that is a concatenation of input nodes.
    """
    return Concat(NodeVector(nodes), axis)


@nameable_op
def softmax(node, axes):  # type: (Node, Iterable[int]) -> Node
    """Softmax operation on input tensor."""
    if type(axes) is not set:
        axes = set(axes)
    return Softmax(node, AxisSet(axes))
