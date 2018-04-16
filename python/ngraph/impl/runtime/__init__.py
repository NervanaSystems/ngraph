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
# flake8: noqa

import sys
import six

# workaround to load the libngraph.so with RTLD_GLOBAL
if six.PY3:
    import os
    flags = os.RTLD_NOW | os.RTLD_GLOBAL
else:
    import ctypes
    flags = sys.getdlopenflags() | ctypes.RTLD_GLOBAL
sys.setdlopenflags(flags)

from _pyngraph.runtime import Backend
from _pyngraph.runtime import TensorView
