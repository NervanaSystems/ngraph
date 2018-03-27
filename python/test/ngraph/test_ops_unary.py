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
import numpy as np
import pytest

import ngraph as ng
from test.ngraph.util import run_op_numeric_data, run_op_node


@pytest.mark.parametrize('ng_api_fn, numpy_fn, input_data', [
    (ng.absolute, np.abs, -1 + np.random.rand(2, 3, 4) * 2),
    (ng.absolute, np.abs, np.float32(-3)),
    (ng.abs, np.abs, -1 + np.random.rand(2, 3, 4) * 2),
    (ng.abs, np.abs, np.float32(-3)),
    (ng.acos, np.arccos, -1 + np.random.rand(2, 3, 4) * 2),
    (ng.acos, np.arccos, np.float32(-0.5)),
    (ng.asin, np.arcsin, -1 + np.random.rand(2, 3, 4) * 2),
    (ng.asin, np.arcsin, np.float32(-0.5)),
    (ng.atan, np.arctan, -100 + np.random.rand(2, 3, 4) * 200),
    (ng.atan, np.arctan, np.float32(-0.5)),
    (ng.ceiling, np.ceil, -100 + np.random.rand(2, 3, 4) * 200),
    (ng.ceiling, np.ceil, np.float32(1.5)),
    (ng.ceil, np.ceil, -100 + np.random.rand(2, 3, 4) * 200),
    (ng.ceil, np.ceil, np.float32(1.5)),
])
def test_unary_op(ng_api_fn, numpy_fn, input_data):
    expected = numpy_fn(input_data)

    result = run_op_node(input_data, ng_api_fn)
    assert np.allclose(result, expected)

    result = run_op_numeric_data(input_data, ng_api_fn)
    assert np.allclose(result, expected)
