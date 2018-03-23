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


def _run_unary_op_node(input_data, unary_op, as_node=True):
    runtime = _get_runtime()
    parameter_a = ng.parameter(input_data.shape, name='A', dtype=np.float32)
    if as_node:
        node = unary_op(parameter_a)
    else:
        # unary_op can handle scalars and numpy.ndarray parameters
        node = unary_op(input_data)
    computation = runtime.computation(node, parameter_a)
    return computation(input_data)


@pytest.mark.parametrize('input_data, as_node', [
    (-1 + np.random.rand(2, 3, 4) * 2, True),
    (-1 + np.random.rand(2, 3, 4) * 2, False),
    (np.float32(-3), False),
])
def test_absolute(input_data, as_node):
    result = _run_unary_op_node(input_data, ng.absolute, as_node)
    expected = np.abs(input_data)
    assert np.allclose(result, expected)


@pytest.mark.parametrize('input_data, as_node', [
    (-1 + np.random.rand(2, 3, 4) * 2, True),
    (-1 + np.random.rand(2, 3, 4) * 2, False),
    (np.float32(-0.5), False),
])
def test_acos(input_data, as_node):
    result = _run_unary_op_node(input_data, ng.acos, as_node)
    expected = np.arccos(input_data)
    assert np.allclose(result, expected)
