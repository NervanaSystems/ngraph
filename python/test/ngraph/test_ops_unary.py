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


def _get_runtime():
    manager_name = pytest.config.getoption('backend', default='CPU')
    return ng.runtime(manager_name=manager_name)


def _run_unary_op_node(input_data, unary_op):
    runtime = _get_runtime()
    parameter_a = ng.parameter(input_data.shape, name='A', dtype=np.float32)
    node = unary_op(parameter_a)
    computation = runtime.computation(node, parameter_a)
    return computation(input_data)


def _run_unary_op_numeric_data(input_data, unary_op):
    runtime = _get_runtime()
    node = unary_op(input_data)
    computation = runtime.computation(node)
    return computation()


@pytest.mark.parametrize('ng_api_fn, numpy_fn, input_data', [
    (ng.absolute, np.abs, -1 + np.random.rand(2, 3, 4) * 2),
    (ng.absolute, np.abs, np.float32(-3)),
    (ng.acos, np.arccos, -1 + np.random.rand(2, 3, 4) * 2),
    (ng.acos, np.arccos, np.float32(-0.5)),
])
def test_unary_op(ng_api_fn, numpy_fn, input_data):
    expected = numpy_fn(input_data)

    result = _run_unary_op_node(input_data, ng_api_fn)
    assert np.allclose(result, expected)

    result = _run_unary_op_numeric_data(input_data, ng_api_fn)
    assert np.allclose(result, expected)
