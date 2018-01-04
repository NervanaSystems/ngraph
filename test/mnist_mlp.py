#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
import numpy as np

import pyngraph.util as util
import pyngraph.runtime.utils as utils
from pyngraph import Float32, Int32, Function, TensorViewType
from pyngraph.op import Parameter, Maximum, Reshape, Dot, Broadcast
from pyngraph.op import Float32Constant, Exp, Log, Sum
from pyngraph.op import Greater, Convert, Reduce
from pyngraph.op import OneHot


float_element_type = Float32.element_type()
int_element_type = Int32.element_type()
bz = 53
lr = 0.2

Input = Parameter(float_element_type, [bz, 28, 28])
Label = Parameter(int_element_type, [bz])
LabelOneHot = Convert((OneHot(Label, [bz, 10], 1)), float_element_type)

MaxParam1 = Parameter(float_element_type, [])
MaxParam2 = Parameter(float_element_type, [])
MaxOutput = TensorViewType(float_element_type, [])
MaxFn = Function(Maximum(MaxParam1, MaxParam2),
                 MaxOutput,
                 [MaxParam1, MaxParam2],
                 'mnist')


def makeScalarConstant(scalar, shape=[], axis_set={}):
    constant_tensor = utils.make_tensor_float32([])
    constant_tensor.write(util.numpy_to_c(np.array([scalar], dtype=np.float32)), 0, 4)
    constant_op = Float32Constant([], constant_tensor)
    constant_broadcast = Broadcast(constant_op, shape, axis_set)
    return constant_broadcast


def makeFloat32Constant(scalar, shape=[], axis_set={}):
    return makeScalarConstant(scalar, shape, axis_set)


def makeFloat32ConstantLike(scalar, op):
    v = set()
    shape = op.get_shape()
    for i in range(len(shape)):
        v.add(i)
    return makeFloat32Constant(scalar, shape, v)


def transpose(op, order):
    v = []
    for i in range(len(order)):
        v.append(op.get_shape()[order[i]])
    new_shape = v
    return Reshape(op, order, new_shape)


def relu(op):
    return Maximum(op, makeFloat32ConstantLike(0., op))

# Flatten
X1 = Reshape(Input, [0, 1, 2], [bz, 784])

# Normalize
X2 = X1 / makeFloat32ConstantLike(255., X1)

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
Max = Reduce(Exp, makeFloat32Constant(0., [], set()), MaxFn, {1})
MaxBroadcast = Broadcast(Max, [bz, 10], {1})
Softmax = Exp / MaxBroadcast

# Loss
LogSoftmax = Log(Softmax)
Loss = Sum(LogSoftmax * LabelOneHot, {0, 1}) / makeFloat32Constant(float(bz), [], set())

# Derivatives
dLogits = Softmax - LabelOneHot
dX5 = dLogits

dX4 = Dot(dX5, transpose(W2, [1, 0]))
dW2 = Dot(transpose(X4, [1, 0]), dX5)
db2 = Sum(dX5, {0})

dX3 = Convert((Greater(X3, makeFloat32Constant(0., [bz, 100], {0, 1}))), float_element_type) * dX4
dX2 = Dot(dX3, transpose(W1, [1, 0]))
dW1 = Dot(transpose(X2, [1, 0]), dX3)
db1 = Sum(dX3, {0})

nW1 = W1 - makeFloat32ConstantLike(lr, dW1) * dW1
nb1 = b1 - makeFloat32ConstantLike(lr, db1) * db1
nW2 = W2 - makeFloat32ConstantLike(lr, dW2) * dW2
nb2 = b2 - makeFloat32ConstantLike(lr, db2) * db2
