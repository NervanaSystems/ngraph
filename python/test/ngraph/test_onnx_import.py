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

import os
import numpy as np

from ngraph.impl.onnx_import import load_onnx_model_file
from test.ngraph.util import get_runtime


def test_import_onnx_function():
    dtype = np.float32
    cur_dir = os.path.dirname(__file__)
    model_path = os.path.join(cur_dir, 'models/add_abc.onnx')
    ng_function = load_onnx_model_file(model_path)[0]

    value_a = np.array([1.0], dtype=dtype)
    value_b = np.array([2.0], dtype=dtype)
    value_c = np.array([3.0], dtype=dtype)

    result = ng_function(value_a, value_b, value_c, runtime=get_runtime())

    assert np.allclose(result, np.array([6], dtype=dtype))
