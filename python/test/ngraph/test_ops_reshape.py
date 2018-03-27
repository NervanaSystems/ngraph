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


def _run_op_node(input_data, op_fun, *args):
    runtime = _get_runtime()
    parameter_a = ng.parameter(input_data.shape, name='A', dtype=np.float32)
    node = op_fun(parameter_a, *args)
    computation = runtime.computation(node, parameter_a)
    return computation(input_data)


def test_broadcast():
    input_data = np.array([1, 2, 3])

    new_shape = [3, 3]
    expected = [[1, 2, 3],
                [1, 2, 3],
                [1, 2, 3]]
    result = _run_op_node(input_data, ng.broadcast, new_shape)
    np.testing.assert_array_equal(result, expected)

    axis = 0
    expected = [[1, 1, 1],
                [2, 2, 2],
                [3, 3, 3]]

    result = _run_op_node(input_data, ng.broadcast, new_shape, axis)
    np.testing.assert_array_equal(result, expected)

    input_data = np.arange(4)
    new_shape = [3, 4, 2, 4]
    expected = np.broadcast_to(input_data, new_shape)
    result = _run_op_node(input_data, ng.broadcast, new_shape)
    np.testing.assert_array_equal(result, expected)
