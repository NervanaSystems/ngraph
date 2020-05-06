# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
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
from test.ngraph.util import run_op_node, get_runtime


@pytest.mark.parametrize('ng_api_helper, numpy_function, reduction_axes', [
    (ng.reduce_max, np.max, [0, 1, 2, 3]),
    (ng.reduce_min, np.min, [0, 1, 2, 3]),
    (ng.reduce_sum, np.sum, [0, 1, 2, 3]),
    (ng.reduce_prod, np.prod, [0, 1, 2, 3]),
    (ng.reduce_max, np.max, [0]),
    (ng.reduce_min, np.min, [0]),
    (ng.reduce_sum, np.sum, [0]),
    (ng.reduce_prod, np.prod, [0]),
    (ng.reduce_max, np.max, [0, 2]),
    (ng.reduce_min, np.min, [0, 2]),
    (ng.reduce_sum, np.sum, [0, 2]),
    (ng.reduce_prod, np.prod, [0, 2]),
])
@pytest.mark.skip_on_gpu
def test_reduction_ops(ng_api_helper, numpy_function, reduction_axes):
    shape = [2, 4, 3, 2]
    np.random.seed(133391)
    input_data = np.random.randn(*shape).astype(np.float32)

    expected = numpy_function(input_data, axis=tuple(reduction_axes))
    result = run_op_node([input_data, reduction_axes], ng_api_helper)
    assert np.allclose(result, expected)


@pytest.mark.parametrize('ng_api_helper, numpy_function, reduction_axes', [
    (ng.reduce_logical_and, np.logical_and.reduce, [0]),
    (ng.reduce_logical_or, np.logical_or.reduce, [0]),
    (ng.reduce_logical_and, np.logical_and.reduce, [0, 2]),
    (ng.reduce_logical_or, np.logical_or.reduce, [0, 2]),
    (ng.reduce_logical_and, np.logical_and.reduce, [0, 1, 2, 3]),
    (ng.reduce_logical_or, np.logical_or.reduce, [0, 1, 2, 3]),
])
@pytest.mark.skip_on_interpreter
def test_reduction_logical_ops(ng_api_helper, numpy_function, reduction_axes):
    shape = [2, 4, 3, 2]
    np.random.seed(133391)
    input_data = np.random.randn(*shape).astype(np.bool)

    expected = numpy_function(input_data, axis=tuple(reduction_axes))
    result = run_op_node([input_data, reduction_axes], ng_api_helper)
    assert np.allclose(result, expected)


@pytest.mark.skip_on_gpu
def test_argmax():
    runtime = get_runtime()
    input_x = ng.constant(np.array([[9, 2, 10],
                                    [12, 8, 4],
                                    [6, 1, 5],
                                    [3, 11, 7]], dtype=np.float32))
    model = runtime.computation(ng.argmax(input_x, 0))
    result = model()
    assert np.allclose(result,
                       np.array([1, 3, 0], dtype=np.int32))


@pytest.mark.skip_on_gpu
def test_argmin():
    runtime = get_runtime()
    input_x = ng.constant(np.array([[12, 2, 10],
                                    [9, 8, 4],
                                    [6, 1, 5],
                                    [3, 11, 7]], dtype=np.float32))
    model = runtime.computation(ng.argmin(input_x, 0))
    result = model()
    assert np.allclose(result,
                       np.array([3, 2, 1], dtype=np.int32))


@pytest.mark.skip_on_gpu
def test_topk():
    runtime = get_runtime()
    input_x = ng.constant(np.array([[9, 2, 10],
                                    [12, 8, 4],
                                    [6, 1, 5],
                                    [3, 11, 7]], dtype=np.float32))
    K = ng.constant(4)
    comp_topk = ng.topk(input_x, K, 0, 'max', 'value')

    model0 = runtime.computation(ng.get_output_element(comp_topk, 0))
    result0 = model0()
    assert np.allclose(result0,
                       np.array([[12, 11, 10],
                                 [9, 8, 7],
                                 [6, 2, 5],
                                 [3, 1, 4]], dtype=np.float32))

    model1 = runtime.computation(ng.get_output_element(comp_topk, 1))
    result1 = model1()
    assert np.allclose(result1,
                       np.array([[1, 3, 0],
                                 [0, 1, 3],
                                 [2, 0, 2],
                                 [3, 2, 1]], dtype=np.int32))


@pytest.mark.parametrize('ng_api_helper, numpy_function, reduction_axes', [
    (ng.reduce_mean, np.mean, [0, 1, 2, 3]),
    (ng.reduce_mean, np.mean, [0]),
    (ng.reduce_mean, np.mean, [0, 2]),
])
@pytest.mark.skip_on_gpu
@pytest.mark.skip_on_cpu
@pytest.mark.skip_on_interpreter
@pytest.mark.skip_on_intelgpu
def test_reduce_mean_op(ng_api_helper, numpy_function, reduction_axes):
    shape = [2, 4, 3, 2]
    np.random.seed(133391)
    input_data = np.random.randn(*shape).astype(np.float32)

    expected = numpy_function(input_data, axis=tuple(reduction_axes))
    result = run_op_node([input_data, reduction_axes], ng_api_helper)
    assert np.allclose(result, expected)


@pytest.mark.parametrize('input_shape, cumsum_axis, reverse', [
    ([5, 2], 0, False),
    ([5, 2], 1, False),
    ([5, 2, 6], 2, False),
    ([5, 2], 0, True),
])
def test_cum_sum(input_shape, cumsum_axis, reverse):
    input_data = np.arange(np.prod(input_shape)).reshape(input_shape)

    if reverse:
        expected = np.cumsum(input_data[::-1], axis=cumsum_axis)[::-1]
    else:
        expected = np.cumsum(input_data, axis=cumsum_axis)

    runtime = get_runtime()
    node = ng.cum_sum(input_data, cumsum_axis, reverse=reverse)
    computation = runtime.computation(node)
    result = computation()
    assert np.allclose(result, expected)


def test_normalize_l2():
    input_shape = [1, 2, 3, 4]
    input_data = np.arange(np.prod(input_shape)).reshape(input_shape).astype(np.float32)
    input_data += 1
    axes = np.array([1, 2, 3]).astype(np.int64)
    eps = 1e-6
    eps_mode = 'add'

    runtime = get_runtime()
    node = ng.normalize_l2(input_data, axes, eps, eps_mode)
    computation = runtime.computation(node)
    result = computation()

    expected = np.array([0.01428571, 0.02857143, 0.04285714, 0.05714286, 0.07142857, 0.08571429,
                         0.1, 0.11428571, 0.12857144, 0.14285715, 0.15714286, 0.17142858,
                         0.18571429, 0.2, 0.21428572, 0.22857143, 0.24285714, 0.25714287,
                         0.27142859, 0.2857143, 0.30000001, 0.31428573, 0.32857144, 0.34285715, ]
                        ).reshape(input_shape)

    assert np.allclose(result, expected)
