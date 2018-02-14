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

from ngraph_api.ops import absolute
from ngraph_api.ops import absolute as abs
from ngraph_api.ops import add
from ngraph_api.ops import broadcast
from ngraph_api.ops import ceiling
from ngraph_api.ops import ceiling as ceil
from ngraph_api.ops import constant
from ngraph_api.ops import divide
from ngraph_api.ops import equal
from ngraph_api.ops import exp
from ngraph_api.ops import floor
from ngraph_api.ops import greater
from ngraph_api.ops import greater_eq
from ngraph_api.ops import log
from ngraph_api.ops import less
from ngraph_api.ops import less_eq
from ngraph_api.ops import maximum
from ngraph_api.ops import minimum
from ngraph_api.ops import multiply
from ngraph_api.ops import negative
from ngraph_api.ops import not_equal
from ngraph_api.ops import parameter
from ngraph_api.ops import sqrt
from ngraph_api.ops import subtract
from ngraph_api.ops import tanh

from ngraph_api.runtime import runtime
