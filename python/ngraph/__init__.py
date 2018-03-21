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
"""ngraph module namespace, exposing factory functions for all ops and other classes."""

from ngraph.ops import absolute
from ngraph.ops import absolute as abs
from ngraph.ops import add
from ngraph.ops import avg_pool
from ngraph.ops import broadcast
from ngraph.ops import ceiling
from ngraph.ops import ceiling as ceil
from ngraph.ops import concat
from ngraph.ops import constant
from ngraph.ops import convert
from ngraph.ops import convolution
from ngraph.ops import divide
from ngraph.ops import dot
from ngraph.ops import equal
from ngraph.ops import exp
from ngraph.ops import floor
from ngraph.ops import greater
from ngraph.ops import greater_eq
from ngraph.ops import log
from ngraph.ops import less
from ngraph.ops import less_eq
from ngraph.ops import logical_not
from ngraph.ops import max
from ngraph.ops import maximum
from ngraph.ops import max_pool
from ngraph.ops import min
from ngraph.ops import minimum
from ngraph.ops import multiply
from ngraph.ops import negative
from ngraph.ops import not_equal
from ngraph.ops import parameter
from ngraph.ops import prod
from ngraph.ops import reshape
from ngraph.ops import slice
from ngraph.ops import sqrt
from ngraph.ops import subtract
from ngraph.ops import sum
from ngraph.ops import tanh

from ngraph.runtime import runtime
