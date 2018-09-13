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
from test.ngraph.util import run_op_node, get_runtime


@pytest.mark.parametrize('ng_api_helper, numpy_function, reduction_axes', [
    (ng.max, np.max, None),
    (ng.min, np.min, None),
    (ng.sum, np.sum, None),
    (ng.prod, np.prod, None),
    (ng.max, np.max, (0, )),
    (ng.min, np.min, (0, )),
    (ng.sum, np.sum, (0, )),
    (ng.prod, np.prod, (0, )),
    (ng.max, np.max, (0, 2)),
    (ng.min, np.min, (0, 2)),
    (ng.sum, np.sum, (0, 2)),
    (ng.prod, np.prod, (0, 2)),
])
@pytest.config.gpu_skip(reason='Not implemented')
def test_reduction_ops(ng_api_helper, numpy_function, reduction_axes):
    shape = [2, 4, 3, 2]
    np.random.seed(133391)
    input_data = np.random.randn(*shape).astype(np.float32)

    expected = numpy_function(input_data, axis=reduction_axes)
    result = run_op_node([input_data], ng_api_helper, reduction_axes)
    assert np.allclose(result, expected)


@pytest.config.gpu_skip(reason='Not implemented')
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


@pytest.config.gpu_skip(reason='Not implemented')
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


@pytest.config.gpu_skip(reason='Not implemented')
def test_topk():
    runtime = get_runtime()
    input_x = ng.constant(np.array([[9, 2, 10],
                                    [12, 8, 4],
                                    [6, 1, 5],
                                    [3, 11, 7]], dtype=np.float32))
    comp_topk = ng.topk(input_x, 4, 0)
    model0 = runtime.computation(ng.get_output_element(comp_topk, 0))
    result0 = model0()
    assert np.allclose(result0,
                       np.array([[1, 3, 0],
                                 [0, 1, 3],
                                 [2, 0, 2],
                                 [3, 2, 1]], dtype=np.int32))
    model1 = runtime.computation(ng.get_output_element(comp_topk, 1))
    result1 = model1()
    assert np.allclose(result1,
                       np.array([[12, 11, 10],
                                 [9, 8, 7],
                                 [6, 2, 5],
                                 [3, 1, 4]], dtype=np.float32))


def test_reduce():
    from functools import reduce
    np.random.seed(133391)

    # default reduce all axes
    init_val = np.float32(0.)
    input_data = np.random.randn(3, 4, 5).astype(np.float32)
    expected = np.sum(input_data)
    reduction_function_args = [init_val, ng.impl.op.Add]
    result = run_op_node([input_data], ng.reduce, *reduction_function_args)
    assert np.allclose(result, expected)

    reduction_axes = (0, 2)
    init_val = np.float32(0.)
    input_data = np.random.randn(3, 4, 5).astype(np.float32)
    expected = np.sum(input_data, axis=reduction_axes)
    reduction_function_args = [init_val, ng.impl.op.Add, list(reduction_axes)]
    result = run_op_node([input_data], ng.reduce, *reduction_function_args)
    assert np.allclose(result, expected)

    reduction_axes = (0, )
    input_data = np.random.randn(100).astype(np.float32)
    expected = reduce(lambda x, y: x - y, input_data, np.float32(0.))
    reduction_function_args = [init_val, ng.impl.op.Subtract, list(reduction_axes)]
    result = run_op_node([input_data], ng.reduce, *reduction_function_args)
    assert np.allclose(result, expected)
