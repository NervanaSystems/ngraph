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

import nwrapper.ngraph.types.TraitedType as TraitedType
import nwrapper.ngraph.ops.Parameter as Parameter
import nwrapper.ngraph.types.TensorViewType as TensorViewType
import nwrapper.ngraph.Function as Function
import nwrapper.ngraph.ops.Maximum as Maximum
import nwrapper.ngraph.ops.Reshape as Reshape
import nwrapper.ngraph.ops.Dot as Dot
import nwrapper.ngraph.ops.Broadcast as Broadcast
import nwrapper.ngraph.runtime.Utils as Utils
import nwrapper.ngraph.ops.ParameterizedConstant as ParameterizedConstant
import nwrapper.ngraph.ops.Exp as Exp
import nwrapper.ngraph.ops.Log as Log
import nwrapper.ngraph.ops.Sum as Sum
import nwrapper.ngraph.ops.Greater as Greater
import nwrapper.ngraph.ops.Convert as Convert
import nwrapper.ngraph.ops.Reduce as Reduce
import nwrapper.ngraph.Util as Util

float_element_type = TraitedType.TraitedTypeF.element_type()
int_element_type = TraitedType.TraitedTypeI.element_type()
bz = 53
lr = 0.2

Input = Parameter.Parameter(float_element_type, [bz, 28, 28])
Label = Parameter.Parameter(int_element_type, [bz])
LabelOneHot = Parameter.Parameter(float_element_type, [bz, 10])

MaxParam1 = Parameter.Parameter(float_element_type, [])
MaxParam2 = Parameter.Parameter(float_element_type, [])
MaxOutput = TensorViewType.TensorViewType(float_element_type, []) 
MaxFn = Function.Function(Maximum.Maximum(MaxParam1, MaxParam2), MaxOutput, [MaxParam1, MaxParam2], 'mnist')

def makeScalarConstant(scalar, shape=[], axis_set={}):
    constant_tensor = Utils.make_tensor([])
    constant_tensor.write(Util.numpy_to_c(np.array([scalar], dtype=np.float32)), 0, 4)
    constant_op = ParameterizedConstant.ParameterizedConstantF([], constant_tensor)
    constant_broadcast = Broadcast.Broadcast(constant_op, shape, axis_set)
    return constant_broadcast

def makeFloat32Constant(scalar, shape=[], axis_set={}):
    return makeScalarConstant(scalar, shape, axis_set)

def makeFloat32ConstantLike(scalar, op):
    v = set()
    shape = op.get_shape()
    for i in range (len(shape)):
        v.add(i)
    return makeFloat32Constant(scalar, shape, v)

def transpose(op, order):
    v = []
    for i in range (len(order)):
        v.append(op.get_shape()[order[i]])    
    new_shape = v
    return Reshape.Reshape(op, order, new_shape)    

def relu(op):
    return Maximum.Maximum(op, makeFloat32ConstantLike(0., op))  

# Flatten
X1 = Reshape.Reshape(Input, [0, 1, 2], [bz, 784])

# Normalize
X2 = X1 / makeFloat32ConstantLike(255., X1) 

# Affine 1
W1 = Parameter.Parameter(float_element_type, [784, 100])
b1 = Parameter.Parameter(float_element_type, [100])
X3 = Dot.Dot(X2, W1) + Broadcast.Broadcast(b1, [bz, 100], {0}) 
X4 = relu(X3)

#Affine 2
W2 = Parameter.Parameter(float_element_type, [100, 10])
b2 = Parameter.Parameter(float_element_type, [10])
X5 = Dot.Dot(X4, W2) + Broadcast.Broadcast(b2, [bz, 10], {0})

# Softmax
Logits = X5
Exp = Exp.Exp(Logits) 
Max = Reduce.Reduce(Exp, makeFloat32Constant(0., [], set()), MaxFn, {1})
MaxBroadcast = Broadcast.Broadcast(Max, [bz, 10], {1})
Softmax = Exp / MaxBroadcast

# Loss
LogSoftmax = Log.Log(Softmax)
Loss = Sum.Sum(LogSoftmax * LabelOneHot, {0, 1})/makeFloat32Constant(float(bz), [], set())

# Derivatives
dLogits = Softmax - LabelOneHot
dX5 = dLogits

dX4 = Dot.Dot(dX5, transpose(W2, [1, 0]))
dW2 = Dot.Dot(transpose(X4, [1, 0]), dX5)
db2 = Sum.Sum(dX5, {0})

dX3 = Convert.Convert((Greater.Greater(X3, makeFloat32Constant(0., [bz, 100], {0, 1}))), float_element_type) * dX4
dX2 = Dot.Dot(dX3, transpose(W1, [1, 0]))
dW1 = Dot.Dot(transpose(X2, [1, 0]), dX3)
db1 = Sum.Sum(dX3, {0})

nW1 = W1 - makeFloat32ConstantLike(lr, dW1) * dW1
nb1 = b1 - makeFloat32ConstantLike(lr, db1) * db1
nW2 = W2 - makeFloat32ConstantLike(lr, dW2) * dW2
nb2 = b2 - makeFloat32ConstantLike(lr, db2) * db2
