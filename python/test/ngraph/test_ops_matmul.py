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
from test.ngraph.util import run_op_node


@pytest.mark.parametrize('left_shape, right_shape, reduction_axes_count, numpy_axes', [
    # matrix, vector
    ([2, 4], [4], None, 1),
    ([4], [4, 2], None, 1),
    # matrix, matrix
    ([2, 4], [4, 2], None, 1),
    # result is a scalar
    ([2, 4], [2, 4], 2, 2),
    # tensor, vector
    ([2, 4, 5], [5], None, 1),
    ([5], [5, 4, 2], None, 1),
    # tensor, matrix
    ([2, 4, 5], [5, 4], None, 1),
    ([5, 4], [4, 5, 2], None, 1),
    # tensor, tensor
    ([2, 3, 4, 5], [5, 2, 3], None, 1),
    ([2, 3, 4, 5], [4, 5, 2, 4], 2, 2),
])
@pytest.config.gpu_skip(reason='under investigation, runtime error is: function failed to compile')
def test_dot(left_shape, right_shape, reduction_axes_count, numpy_axes):
    np.random.seed(133391)
    left_input = -100.0 + np.random.rand(*left_shape) * 200.0
    right_input = -100.0 + np.random.rand(*right_shape) * 200.0

    expected = np.tensordot(left_input, right_input, numpy_axes)
    result = run_op_node([left_input, right_input], ng.dot, reduction_axes_count)
    assert np.allclose(result, expected)


@pytest.config.gpu_skip(reason='under investigation, runtime error is: function failed to compile')
def test_dot_tensor_scalar():
    np.random.seed(133391)
    left_input = 10.0
    right_input = -100.0 + np.random.rand(2, 3, 4) * 200.0
    expected = left_input * right_input

    result = run_op_node([left_input, right_input], ng.dot)
    assert np.allclose(result, expected)

    result = run_op_node([right_input, left_input], ng.dot)
    assert np.allclose(result, expected)
