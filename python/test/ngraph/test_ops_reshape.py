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
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize('val_type, value', [
    (bool, False),
    (bool, np.empty((2, 2), dtype=bool)),
    (np.float32, np.float32(0.1234)),
    (np.float32, -1 + np.random.rand(2, 3, 4) * 2),
    (np.float64, np.float64(0.1234)),
    (np.float64, np.array(-1 + np.random.rand(2, 3, 4) * 2, dtype=np.float64)),
    (np.int8, np.int8(-63)),
    (np.int8, np.random.randint(-8, 8, size=(2, 2), dtype=np.int8)),
    (np.int16, np.int16(-12345)),
    (np.int16, np.random.randint(-64, 64, size=(2, 2), dtype=np.int16)),
    (np.int32, np.int32(-123456)),
    (np.int32, np.random.randint(-1024, 1024, size=(2, 2), dtype=np.int32)),
    (np.int64, np.int64(-1234567)),
    (np.int64, np.random.randint(-16383, 16383, size=(2, 2), dtype=np.int64)),
    (np.uint8, np.uint8(63)),
    (np.uint8, np.random.randint(0, 8, size=(2, 2), dtype=np.uint8)),
    (np.uint16, np.uint16(12345)),
    (np.uint16, np.random.randint(0, 64, size=(2, 2), dtype=np.uint16)),
    (np.uint32, np.uint32(123456)),
    (np.uint32, np.random.randint(0, 1024, size=(2, 2), dtype=np.uint32)),
    (np.uint64, np.uint64(1234567)),
    (np.uint64, np.random.randint(0, 16383, size=(2, 2), dtype=np.uint64)),
])
def test_constant(val_type, value):
    expected = np.array(value, dtype=val_type)
    result = run_op_numeric_data(value, ng.constant, val_type)
    np.testing.assert_allclose(result, expected)
