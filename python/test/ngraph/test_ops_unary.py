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


@pytest.mark.parametrize('ng_api_fn, numpy_fn, range_start, range_end', [
    (ng.absolute, np.abs, -1, 1),
    (ng.abs, np.abs, -1, 1),
    (ng.acos, np.arccos, -1, 1),
    (ng.asin, np.arcsin, -1, 1),
    (ng.atan, np.arctan, -100, 100),
    (ng.ceiling, np.ceil, -100, 100),
    (ng.ceil, np.ceil, -100, 100),
    (ng.cos, np.cos, -np.pi, np.pi),
    (ng.cosh, np.cosh, -np.pi, np.pi),
    (ng.exp, np.exp, -100, 100),
    (ng.floor, np.floor, -100, 100),
    (ng.log, np.log, 0, 100),
])
def test_unary_op_array(ng_api_fn, numpy_fn, range_start, range_end):
    np.random.seed(133391)
    input_data = range_start + np.random.rand(2, 3, 4) * (range_end - range_start)
    expected = numpy_fn(input_data)

    result = run_op_node([input_data], ng_api_fn)
    assert np.allclose(result, expected)

    result = run_op_numeric_data(input_data, ng_api_fn)
    assert np.allclose(result, expected)


@pytest.mark.parametrize('ng_api_fn, numpy_fn, input_data', [
    (ng.absolute, np.abs, np.float32(-3)),
    (ng.abs, np.abs, np.float32(-3)),
    (ng.acos, np.arccos, np.float32(-0.5)),
    (ng.asin, np.arcsin, np.float32(-0.5)),
    (ng.atan, np.arctan, np.float32(-0.5)),
    (ng.ceiling, np.ceil, np.float32(1.5)),
    (ng.ceil, np.ceil, np.float32(1.5)),
    (ng.cos, np.cos, np.float32(np.pi / 4.0)),
    (ng.cosh, np.cosh, np.float32(np.pi / 4.0)),
    (ng.exp, np.exp, np.float32(1.5)),
    (ng.floor, np.floor, np.float32(1.5)),
    (ng.log, np.log, np.float32(1.5)),
])
def test_unary_op_scalar(ng_api_fn, numpy_fn, input_data):
    expected = numpy_fn(input_data)

    result = run_op_node([input_data], ng_api_fn)
    assert np.allclose(result, expected)

    result = run_op_numeric_data(input_data, ng_api_fn)
    assert np.allclose(result, expected)
