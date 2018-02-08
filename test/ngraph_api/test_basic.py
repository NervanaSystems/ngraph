#*******************************************************************************
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
#*******************************************************************************

import numpy as np
import pytest

import ngraph_api as ng


@pytest.mark.parametrize('dtype', [np.float32, np.float64,
                                   np.int8, np.int16, np.int32, np.int64,
                                   np.uint8, np.uint16, np.uint32, np.uint64])
def test_simple_computation_on_ndarrays(dtype):
    manager_name = pytest.config.getoption('backend', default='INTERPRETER')

    shape = [2, 2]
    parameter_a = ng.parameter(shape, dtype=dtype, name='A')
    parameter_b = ng.parameter(shape, dtype=dtype, name='B')
    parameter_c = ng.parameter(shape, dtype=dtype, name='C')
    model = (parameter_a + parameter_b) * parameter_c
    runtime = ng.runtime(manager_name=manager_name)
    computation = runtime.computation(model, parameter_a, parameter_b, parameter_c)

    value_a = np.array([[1, 2], [3, 4]], dtype=dtype)
    value_b = np.array([[5, 6], [7, 8]], dtype=dtype)
    value_c = np.array([[9, 10], [11, 12]], dtype=dtype)
    result = computation(value_a, value_b, value_c)
    assert np.allclose(result, np.array([[54, 80], [110, 144]], dtype=dtype))
