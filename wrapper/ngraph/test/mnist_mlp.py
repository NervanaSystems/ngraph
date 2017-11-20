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

import wrapper.ngraph.types.clsTraitedType as clsTraitedType
import wrapper.ngraph.ops.clsParameter as clsParameter
import wrapper.ngraph.runtime.clsTensorViewType as clsTensorViewType
import wrapper.ngraph.clsFunction as clsFunction
import wrapper.ngraph.ops.clsMaximum as clsMaximum
import wrapper.ngraph.ops.clsReshape as clsReshape
import wrapper.ngraph.ops.clsDot as clsDot
import wrapper.ngraph.ops.clsBroadcast as clsBroadcast
import wrapper.ngraph.runtime.clsUtils as clsUtils
import wrapper.ngraph.ops.clsParameterizedConstant as clsParameterizedConstant
import wrapper.ngraph.ops.clsExp as clsExp
import wrapper.ngraph.ops.clsReduce as clsReduce

float_element_type = clsTraitedType.TraitedTypeF.element_type()
int_element_type = clsTraitedType.TraitedTypeI.element_type()
bz = 53
Input = clsParameter.Parameter(float_element_type, [bz, 28, 28])
Label = clsParameter.Parameter(int_element_type, [bz])

MaxParam1 = clsParameter.Parameter(float_element_type, [])
MaxParam2 = clsParameter.Parameter(float_element_type, [])
MaxOutput = clsTensorViewType.TensorViewType(float_element_type, []) 
MaxFn = clsFunction.Function(clsMaximum.Maximum(MaxParam1, MaxParam2), MaxOutput, [MaxParam1, MaxParam2], 'mnist')

constant_tensor = clsUtils.make_tensor([]) 
#constant_tensor = [None]*255
constant_op = clsParameterizedConstant.ParameterizedConstantF([], constant_tensor) 
constant_broadcast = clsBroadcast.Broadcast(constant_op, [bz, 784], {0, 1})

# Flatten
X1 = clsReshape.Reshape(Input, [0, 1, 2], [bz, 784])

# Normalize
X2 = X1/constant_broadcast 

# Affine 1
W1 = clsParameter.Parameter(float_element_type, [784, 100])
b1 = clsParameter.Parameter(float_element_type, [100])
X3 = clsDot.Dot(X2, W1) + clsBroadcast.Broadcast(b1, [bz, 100], {0}) 

constant_broadcast_1 = clsBroadcast.Broadcast(constant_op, [bz, 100], {0, 1})

X4 = clsMaximum.Maximum(X3, constant_broadcast_1)

#Affine 2
W2 = clsParameter.Parameter(float_element_type, [100, 10])
b2 = clsParameter.Parameter(float_element_type, [10])
X5 = clsDot.Dot(X4, W2) + clsBroadcast.Broadcast(b2, [bz, 10], {0})

# Softmax and loss
constant_broadcast_2 = clsBroadcast.Broadcast(constant_op, [0], {0})
Logits = X5
Exp = clsExp.Exp(Logits) 
Max = clsReduce.Reduce(Exp, constant_broadcast_2, MaxFn, {1}) 
MaxBroadcast = clsBroadcast.Broadcast(Max, [bz, 10], {1})
Softmax = Exp / MaxBroadcast
