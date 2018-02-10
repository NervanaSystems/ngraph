#!/usr/bin/env python
# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
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
from pyngraph import Type, Function
from pyngraph import Node
from pyngraph.op import Parameter, Maximum, Reshape, Dot, Broadcast
from pyngraph.op import Constant, Exp, Log, Sum
from pyngraph.op import Greater, Convert, Reduce
from pyngraph.op import OneHot

from typing import List, Dict, Set


float_element_type = Type.f32
int_element_type = Type.i32
bz = 53
lr = 0.2

Input = Parameter(float_element_type, [bz, 28, 28])
Label = Parameter(int_element_type, [bz])
LabelOneHot = Convert((OneHot(Label, [bz, 10], 1)), float_element_type)

MaxParam1 = Parameter(float_element_type, [])
MaxParam2 = Parameter(float_element_type, [])
MaxFn = Function([Maximum(MaxParam1, MaxParam2)],
                 [MaxParam1, MaxParam2],
                 'mnist')


def make_scalar_constant(elem_type, scalar, shape=None, axis_set=None):
    # type: (int, float, List[int], Set[int]) -> float
    """Create a Constant node for scalar value."""
    if shape is None:
        shape = []
    if axis_set is None:
        axis_set = set()
    scalar_shape = []  # type: List[int]
    constant_op = Constant(elem_type, scalar_shape, [scalar])
    constant_broadcast = Broadcast(constant_op, shape, axis_set)
    return constant_broadcast


def make_float32_constant(scalar, shape=None, axis_set=None):
    # type: (float, List[int], Set[int]) -> float
    """Create a Constant node for float value."""
    if shape is None:
        shape = []
    if axis_set is None:
        axis_set = set()
    return make_scalar_constant(Type.f32, scalar, shape, axis_set)


def make_float32_constant_like(scalar, op):  # type: (float, Node) -> float
    """Create a Constant node for float value."""
    v = set()
    shape = op.get_shape()
    for i in range(len(shape)):
        v.add(i)
    return make_float32_constant(scalar, shape, v)


def transpose(op, order):  # type: (Node, List[int]) -> Node
    """Transpose data via reshape."""
    v = []
    for i in range(len(order)):
        v.append(op.get_shape()[order[i]])
    new_shape = v
    return Reshape(op, order, new_shape)


def relu(op):  # type: (Node) -> Node
    """Relu operator."""
    return Maximum(op, make_float32_constant_like(0., op))


# Flatten
X1 = Reshape(Input, [0, 1, 2], [bz, 784])

# Normalize
X2 = X1 / make_float32_constant_like(255., X1)

# Affine 1
W1 = Parameter(float_element_type, [784, 100])
b1 = Parameter(float_element_type, [100])
X3 = Dot(X2, W1) + Broadcast(b1, [bz, 100], {0})
X4 = relu(X3)

# Affine 2
W2 = Parameter(float_element_type, [100, 10])
b2 = Parameter(float_element_type, [10])
X5 = Dot(X4, W2) + Broadcast(b2, [bz, 10], {0})

# Softmax
Logits = X5
Exp = Exp(Logits)
Max = Reduce(Exp, make_float32_constant(0., [], set()), MaxFn, {1})
MaxBroadcast = Broadcast(Max, [bz, 10], {1})
Softmax = Exp / MaxBroadcast

# Loss
LogSoftmax = Log(Softmax)
Loss = Sum(LogSoftmax * LabelOneHot, {0, 1}) / make_float32_constant(float(bz), [], set())

# Derivatives
dLogits = Softmax - LabelOneHot
dX5 = dLogits

dX4 = Dot(dX5, transpose(W2, [1, 0]))
dW2 = Dot(transpose(X4, [1, 0]), dX5)
db2 = Sum(dX5, {0})

dX3 = Convert((Greater(X3, make_float32_constant(0., [bz, 100], {0, 1}))), float_element_type) * dX4
dX2 = Dot(dX3, transpose(W1, [1, 0]))
dW1 = Dot(transpose(X2, [1, 0]), dX3)
db1 = Sum(dX3, {0})

nW1 = W1 - make_float32_constant_like(lr, dW1) * dW1
nb1 = b1 - make_float32_constant_like(lr, db1) * db1
nW2 = W2 - make_float32_constant_like(lr, dW2) * dW2
nb2 = b2 - make_float32_constant_like(lr, db2) * db2
