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

import ngraph_api as ng


@pytest.mark.parametrize('ng_api_helper,numpy_function', [
    (ng.add, np.add),
    (ng.divide, np.divide),
    (ng.multiply, np.multiply),
    (ng.subtract, np.subtract),
    (ng.equal, np.equal),
    (ng.minimum, np.minimum),
    (ng.maximum, np.maximum),
])
def test_binary_op(ng_api_helper, numpy_function):
    manager_name = pytest.config.getoption('backend', default='INTERPRETER')
    runtime = ng.runtime(manager_name=manager_name)

    shape = [2, 2]
    parameter_a = ng.parameter(shape, name='A', dtype=np.float32)
    parameter_b = ng.parameter(shape, name='B', dtype=np.float32)

    model = ng_api_helper(parameter_a, parameter_b)
    computation = runtime.computation(model, parameter_a, parameter_b)

    value_a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    value_b = np.array([[5, 6], [7, 8]], dtype=np.float32)

    result = computation(value_a, value_b)
    expected = numpy_function(value_a, value_b)
    assert np.allclose(result, expected)


@pytest.mark.parametrize('ng_api_helper,numpy_function', [
    (ng.add, np.add),
    (ng.divide, np.divide),
    (ng.multiply, np.multiply),
    (ng.subtract, np.subtract),
    (ng.equal, np.equal),
    (ng.minimum, np.minimum),
    (ng.maximum, np.maximum),
])
def test_binary_op_with_scalar(ng_api_helper, numpy_function):
    manager_name = pytest.config.getoption('backend', default='INTERPRETER')
    runtime = ng.runtime(manager_name=manager_name)

    value_a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    value_b = np.array([[5, 6], [7, 8]], dtype=np.float32)

    shape = [2, 2]
    parameter_a = ng.parameter(shape, name='A', dtype=np.float32)

    model = ng_api_helper(parameter_a, value_b)
    computation = runtime.computation(model, parameter_a)

    result = computation(value_a)
    expected = numpy_function(value_a, value_b)
    assert np.allclose(result, expected)
