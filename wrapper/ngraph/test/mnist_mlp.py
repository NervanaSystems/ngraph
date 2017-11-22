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

import wrapper.ngraph.types.TraitedType as TraitedType
import wrapper.ngraph.ops.Parameter as Parameter
import wrapper.ngraph.runtime.TensorViewType as TensorViewType
import wrapper.ngraph.Function as Function
import wrapper.ngraph.ops.Maximum as Maximum
import wrapper.ngraph.ops.Reshape as Reshape
import wrapper.ngraph.ops.Dot as Dot
import wrapper.ngraph.ops.Broadcast as Broadcast
import wrapper.ngraph.runtime.Utils as Utils
import wrapper.ngraph.ops.ParameterizedConstant as ParameterizedConstant
import wrapper.ngraph.ops.Exp as Exp
import wrapper.ngraph.ops.Reduce as Reduce

float_element_type = TraitedType.TraitedTypeF.element_type()
int_element_type = TraitedType.TraitedTypeI.element_type()
bz = 53
Input = Parameter.Parameter(float_element_type, [bz, 28, 28])
Label = Parameter.Parameter(int_element_type, [bz])

MaxParam1 = Parameter.Parameter(float_element_type, [])
MaxParam2 = Parameter.Parameter(float_element_type, [])
MaxOutput = TensorViewType.TensorViewType(float_element_type, []) 
MaxFn = Function.Function(Maximum.Maximum(MaxParam1, MaxParam2), MaxOutput, [MaxParam1, MaxParam2], 'mnist')

constant_tensor = Utils.make_tensor([]) 
#constant_tensor = [None]*255
constant_op = ParameterizedConstant.ParameterizedConstantF([], constant_tensor) 
constant_broadcast = Broadcast.Broadcast(constant_op, [bz, 784], {0, 1})

# Flatten
X1 = Reshape.Reshape(Input, [0, 1, 2], [bz, 784])

# Normalize
X2 = X1/constant_broadcast 

# Affine 1
W1 = Parameter.Parameter(float_element_type, [784, 100])
b1 = Parameter.Parameter(float_element_type, [100])
X3 = Dot.Dot(X2, W1) + Broadcast.Broadcast(b1, [bz, 100], {0}) 

constant_broadcast_1 = Broadcast.Broadcast(constant_op, [bz, 100], {0, 1})

X4 = Maximum.Maximum(X3, constant_broadcast_1)

#Affine 2
W2 = Parameter.Parameter(float_element_type, [100, 10])
b2 = Parameter.Parameter(float_element_type, [10])
X5 = Dot.Dot(X4, W2) + Broadcast.Broadcast(b2, [bz, 10], {0})

# Softmax and loss
constant_broadcast_2 = Broadcast.Broadcast(constant_op, [0], {0})
Logits = X5
Exp = Exp.Exp(Logits) 
Max = Reduce.Reduce(Exp, constant_broadcast_2, MaxFn, {1}) 
MaxBroadcast = Broadcast.Broadcast(Max, [bz, 10], {1})
Softmax = Exp / MaxBroadcast
