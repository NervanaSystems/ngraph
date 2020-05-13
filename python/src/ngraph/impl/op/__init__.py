# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
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
if sys.platform != 'win32':
    if six.PY3:
        import os
        flags = os.RTLD_NOW | os.RTLD_GLOBAL
    else:
        import ctypes
        flags = sys.getdlopenflags() | ctypes.RTLD_GLOBAL
    sys.setdlopenflags(flags)

from _pyngraph.op import AllReduce
from _pyngraph.op import ArgMax
from _pyngraph.op import ArgMin
from _pyngraph.op import AvgPool
from _pyngraph.op import AvgPoolBackprop
from _pyngraph.op import BatchNormTraining
from _pyngraph.op import BatchNormInference
from _pyngraph.op import BatchNormTrainingBackprop
from _pyngraph.op import Broadcast
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
from _pyngraph.op import DepthToSpace
from _pyngraph.op import Dequantize
from _pyngraph.op import Dot
from _pyngraph.op import Gelu
from _pyngraph.op import Gemm
from _pyngraph.op import GetOutputElement
from _pyngraph.op import GRN
from _pyngraph.op import GroupConvolution
from _pyngraph.op import HardSigmoid
from _pyngraph.op import Max
from _pyngraph.op import Maximum
from _pyngraph.op import MaxPool
from _pyngraph.op import MaxPoolBackprop
from _pyngraph.op import Min
from _pyngraph.op import MVN
from _pyngraph.op import Op
from _pyngraph.op import Parameter
from _pyngraph.op import Product
from _pyngraph.op import Quantize
from _pyngraph.op import QuantizedConvolution
from _pyngraph.op import QuantizedDot
from _pyngraph.op import ReplaceSlice
from _pyngraph.op import RNNCell
from _pyngraph.op import ScaleShift
from _pyngraph.op import ShuffleChannels
from _pyngraph.op import Slice
from _pyngraph.op import Softmax
from _pyngraph.op import SpaceToDepth
from _pyngraph.op import Unsqueeze
