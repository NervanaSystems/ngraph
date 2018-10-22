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
"""
Package: ngraph.op
Low level wrappers for the nGraph c++ api in ngraph::op.
"""

# flake8: noqa

import sys
import six

import numpy as np

# workaround to load the libngraph.so with RTLD_GLOBAL
if six.PY3:
    import os
    flags = os.RTLD_NOW | os.RTLD_GLOBAL
else:
    import ctypes
    flags = sys.getdlopenflags() | ctypes.RTLD_GLOBAL
sys.setdlopenflags(flags)

from _pyngraph.op import Abs
from _pyngraph.op import Acos
from _pyngraph.op import Add
from _pyngraph.op import AllReduce
from _pyngraph.op import And
from _pyngraph.op import ArgMax
from _pyngraph.op import ArgMin
from _pyngraph.op import Asin
from _pyngraph.op import Atan
from _pyngraph.op import AvgPool
from _pyngraph.op import AvgPoolBackprop
from _pyngraph.op import BatchNormTraining
from _pyngraph.op import BatchNormInference
from _pyngraph.op import BatchNormTrainingBackprop
from _pyngraph.op import Broadcast
from _pyngraph.op import Ceiling
from _pyngraph.op import Concat
from _pyngraph.op import Constant

""" Retrieve Constant inner data.

    Internally uses PyBind11 Numpy's buffer protocol.
    
    :return Numpy array containing internally stored constant data.
"""
Constant.get_data = lambda self: np.array(self, copy=True)

from _pyngraph.op import Convert
from _pyngraph.op import Convolution
from _pyngraph.op import ConvolutionBackpropData
from _pyngraph.op import ConvolutionBackpropFilters
from _pyngraph.op import Cos
from _pyngraph.op import Cosh
from _pyngraph.op import Divide
from _pyngraph.op import Dot
from _pyngraph.op import Equal
from _pyngraph.op import Exp
from _pyngraph.op import Floor
from _pyngraph.op import FunctionCall
from _pyngraph.op import GetOutputElement
from _pyngraph.op import Greater
from _pyngraph.op import GreaterEq
from _pyngraph.op import Less
from _pyngraph.op import LessEq
from _pyngraph.op import Log
from _pyngraph.op import LRN
from _pyngraph.op import Max
from _pyngraph.op import Maximum
from _pyngraph.op import MaxPool
from _pyngraph.op import MaxPoolBackprop
from _pyngraph.op import Min
from _pyngraph.op import Minimum
from _pyngraph.op import Multiply
from _pyngraph.op import Negative
from _pyngraph.op import Not
from _pyngraph.op import NotEqual
from _pyngraph.op import OneHot
from _pyngraph.op import Op
from _pyngraph.op import Or
from _pyngraph.op import Pad
from _pyngraph.op import Parameter
from _pyngraph.op import ParameterVector
from _pyngraph.op import Power
from _pyngraph.op import Product
from _pyngraph.op import Reduce
from _pyngraph.op import Relu
from _pyngraph.op import ReluBackprop
from _pyngraph.op import ReplaceSlice
from _pyngraph.op import Reshape
from _pyngraph.op import Reverse
from _pyngraph.op import Select
from _pyngraph.op import Sign
from _pyngraph.op import Sin
from _pyngraph.op import Sinh
from _pyngraph.op import Slice
from _pyngraph.op import Softmax
from _pyngraph.op import Sqrt
from _pyngraph.op import Subtract
from _pyngraph.op import Sum
from _pyngraph.op import Tan
from _pyngraph.op import Tanh
from _pyngraph.op import TopK
