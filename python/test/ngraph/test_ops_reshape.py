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
from test.ngraph.util import get_runtime, run_op_numeric_data


def test_concat():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6]])
    axis = 0
    expected = np.concatenate((a, b), axis=0)

    runtime = get_runtime()
    parameter_a = ng.parameter(list(a.shape), name='A', dtype=np.float32)
    parameter_b = ng.parameter(list(b.shape), name='B', dtype=np.float32)
    node = ng.concat([parameter_a, parameter_b], axis)
    computation = runtime.computation(node, parameter_a, parameter_b)
    result = computation(a, b)
    assert np.allclose(result, expected)


@pytest.mark.parametrize('val_type, value', [
    (bool, False),
    (bool, np.empty((2, 2), dtype=bool)),
])
def test_constant_from_bool(val_type, value):
    expected = np.array(value, dtype=val_type)
    result = run_op_numeric_data(value, ng.constant, val_type)
    assert np.allclose(result, expected)


@pytest.mark.parametrize('val_type, value', [
    (np.float32, np.float32(0.1234)),
    (np.float64, np.float64(0.1234)),
    (np.int8, np.int8(-63)),
    (np.int16, np.int16(-12345)),
    (np.int32, np.int32(-123456)),
    (np.int64, np.int64(-1234567)),
    (np.uint8, np.uint8(63)),
    (np.uint16, np.uint16(12345)),
    (np.uint32, np.uint32(123456)),
    (np.uint64, np.uint64(1234567)),
])
def test_constant_from_scalar(val_type, value):
    expected = np.array(value, dtype=val_type)
    result = run_op_numeric_data(value, ng.constant, val_type)
    assert np.allclose(result, expected)


@pytest.mark.parametrize('val_type', [
    np.float32,
    np.float64,
])
def test_constant_from_float_array(val_type):
    np.random.seed(133391)
    input_data = np.array(-1 + np.random.rand(2, 3, 4) * 2, dtype=val_type)
    result = run_op_numeric_data(input_data, ng.constant, val_type)
    assert np.allclose(result, input_data)


@pytest.mark.parametrize('val_type, range_start, range_end', [
    (np.int8, -8, 8),
    (np.int16, -64, 64),
    (np.int32, -1024, 1024),
    (np.int64, -16383, 16383),
    (np.uint8, 0, 8),
    (np.uint16, 0, 64),
    (np.uint32, 0, 1024),
    (np.uint64, 0, 16383),
])
def test_constant_from_integer_array(val_type, range_start, range_end):
    np.random.seed(133391)
    input_data = np.array(np.random.randint(range_start, range_end, size=(2, 2)), dtype=val_type)
    result = run_op_numeric_data(input_data, ng.constant, val_type)
    assert np.allclose(result, input_data)
