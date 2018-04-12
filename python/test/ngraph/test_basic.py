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
import json

import ngraph as ng
from test.ngraph.util import get_runtime, run_op_node
from ngraph.impl import Function, NodeVector


@pytest.mark.parametrize('dtype', [np.float32, np.float64,
                                   np.int8, np.int16, np.int32, np.int64,
                                   np.uint8, np.uint16, np.uint32, np.uint64])
@pytest.config.gpu_skip(reason='Not implemented')
def test_simple_computation_on_ndarrays(dtype):
    runtime = get_runtime()

    shape = [2, 2]
    parameter_a = ng.parameter(shape, dtype=dtype, name='A')
    parameter_b = ng.parameter(shape, dtype=dtype, name='B')
    parameter_c = ng.parameter(shape, dtype=dtype, name='C')
    model = (parameter_a + parameter_b) * parameter_c
    computation = runtime.computation(model, parameter_a, parameter_b, parameter_c)

    value_a = np.array([[1, 2], [3, 4]], dtype=dtype)
    value_b = np.array([[5, 6], [7, 8]], dtype=dtype)
    value_c = np.array([[9, 10], [11, 12]], dtype=dtype)
    result = computation(value_a, value_b, value_c)
    assert np.allclose(result, np.array([[54, 80], [110, 144]], dtype=dtype))

    value_a = np.array([[13, 14], [15, 16]], dtype=dtype)
    value_b = np.array([[17, 18], [19, 20]], dtype=dtype)
    value_c = np.array([[21, 22], [23, 24]], dtype=dtype)
    result = computation(value_a, value_b, value_c)
    assert np.allclose(result, np.array([[630, 704], [782, 864]], dtype=dtype))


def test_function_call():
    runtime = get_runtime()
    dtype = int
    shape = [2, 2]
    parameter_a = ng.parameter(shape, dtype=dtype, name='A')
    parameter_b = ng.parameter(shape, dtype=dtype, name='B')
    parameter_c = ng.parameter(shape, dtype=dtype, name='C')
    parameter_list = [parameter_a, parameter_b, parameter_c]
    ops = ((parameter_a + parameter_b) * parameter_c)
    func = Function(NodeVector([ops]), parameter_list, 'addmul')
    fc = ng.function_call(func, NodeVector(parameter_list))
    computation = runtime.computation(fc, parameter_a, parameter_b, parameter_c)

    value_a = np.array([[1, 2], [3, 4]], dtype=dtype)
    value_b = np.array([[5, 6], [7, 8]], dtype=dtype)
    value_c = np.array([[9, 10], [11, 12]], dtype=dtype)
    result = computation(value_a, value_b, value_c)
    assert np.allclose(result, np.array([[54, 80], [110, 144]], dtype=dtype))


def test_serialization():
    dtype = np.float32
    manager_name = pytest.config.getoption('backend', default='CPU')

    shape = [2, 2]
    parameter_a = ng.parameter(shape, dtype=dtype, name='A')
    parameter_b = ng.parameter(shape, dtype=dtype, name='B')
    parameter_c = ng.parameter(shape, dtype=dtype, name='C')
    model = (parameter_a + parameter_b) * parameter_c
    runtime = ng.runtime(manager_name=manager_name)
    computation = runtime.computation(model, parameter_a, parameter_b, parameter_c)
    serialized = computation.serialize(2)
    serial_json = json.loads(serialized)

    assert serial_json[0]['name'] != ''
    assert 10 == len(serial_json[0]['ops'])

    input_data = np.array([1, 2, 3])

    new_shape = [3, 3]
    expected = [[1, 2, 3],
                [1, 2, 3],
                [1, 2, 3]]
    result = run_op_node([input_data], ng.broadcast, new_shape)
    assert np.allclose(result, expected)

    axis = 0
    expected = [[1, 1, 1],
                [2, 2, 2],
                [3, 3, 3]]

    result = run_op_node([input_data], ng.broadcast, new_shape, axis)
    assert np.allclose(result, expected)

    input_data = np.arange(4)
    new_shape = [3, 4, 2, 4]
    expected = np.broadcast_to(input_data, new_shape)
    result = run_op_node([input_data], ng.broadcast, new_shape)
    assert np.allclose(result, expected)


@pytest.mark.parametrize('val_type, input_data', [
    (bool, np.zeros((2, 2), dtype=int)),
])
def test_convert_to_bool(val_type, input_data):
    expected = np.array(input_data, dtype=val_type)
    result = run_op_node([input_data], ng.convert, val_type)
    assert np.allclose(result, expected)


@pytest.mark.parametrize('val_type, range_start, range_end, in_dtype', [
    (np.float32, -8, 8, np.int32),
    (np.float64, -16383, 16383, np.int64),
])
def test_convert_to_float(val_type, range_start, range_end, in_dtype):
    np.random.seed(133391)
    input_data = np.random.randint(range_start, range_end, size=(2, 2), dtype=in_dtype)
    expected = np.array(input_data, dtype=val_type)
    result = run_op_node([input_data], ng.convert, val_type)
    assert np.allclose(result, expected)


@pytest.mark.parametrize('val_type', [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
])
def test_convert_to_int(val_type):
    np.random.seed(133391)
    input_data = np.ceil(-8 + np.random.rand(2, 3, 4) * 16)
    expected = np.array(input_data, dtype=val_type)
    result = run_op_node([input_data], ng.convert, val_type)
    assert np.allclose(result, expected)


@pytest.mark.parametrize('val_type', [
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
])
def test_convert_to_uint(val_type):
    np.random.seed(133391)
    input_data = np.ceil(np.random.rand(2, 3, 4) * 16)
    expected = np.array(input_data, dtype=val_type)
    result = run_op_node([input_data], ng.convert, val_type)
    assert np.allclose(result, expected)
