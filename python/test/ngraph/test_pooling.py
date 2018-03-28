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
from test.ngraph.util import get_runtime


@pytest.fixture
def _ndarray_1x1x4x4():
    return np.arange(11, 27, dtype=np.float32).reshape(1, 1, 4, 4)


def test_avg_pool_2d(_ndarray_1x1x4x4):
    runtime = get_runtime()
    input_data = _ndarray_1x1x4x4
    param = ng.parameter(input_data.shape, name='A', dtype=np.float32)

    window_shape = [2, 2]
    strides = [2, 2]
    expected = [[[[13.5, 15.5],
                  [21.5, 23.5]]]]

    avg_pool_node = ng.avg_pool(param, window_shape, strides)
    computation = runtime.computation(avg_pool_node, param)
    result = computation(input_data)
    assert np.allclose(result, expected)

    expected = [[[[13.5, 14.5, 15.5],
                  [17.5, 18.5, 19.5],
                  [21.5, 22.5, 23.5]]]]
    avg_pool_node = ng.avg_pool(param, window_shape)
    computation = runtime.computation(avg_pool_node, param)
    result = computation(input_data)
    assert np.allclose(result, expected)

    padding_below = [1, 1]
    padding_above = [1, 1]
    strides = [2, 2]
    include_pad = False

    expected = [[[[11.0, 12.5, 14.0],
                  [17.0, 18.5, 20.0],
                  [23.0, 24.5, 26.0]]]]
    avg_pool_node = ng.avg_pool(param, window_shape, strides, padding_below, padding_above,
                                include_pad)
    computation = runtime.computation(avg_pool_node, param)
    result = computation(input_data)
    assert np.allclose(result, expected)

    include_pad = True
    expected = [[[[2.75, 6.25, 3.5],
                  [8.5, 18.5, 10.0],
                  [5.75, 12.25, 6.5]]]]
    avg_pool_node = ng.avg_pool(param, window_shape, strides, padding_below, padding_above,
                                include_pad)
    computation = runtime.computation(avg_pool_node, param)
    result = computation(input_data)
    assert np.allclose(result, expected)


def test_avg_pooling_3d(_ndarray_1x1x4x4):
    rt = get_runtime()
    data = _ndarray_1x1x4x4
    data = np.broadcast_to(data, (1, 1, 4, 4, 4))
    param = ng.parameter(list(data.shape))
    window_shape = [2, 2, 2]
    strides = [2, 2, 2]

    avgpool = ng.avg_pool(param, window_shape, strides)
    comp = rt.computation(avgpool, param)
    result = comp(data)
    result_ref = [[[[[13.5, 15.5],
                     [21.5, 23.5]],

                    [[13.5, 15.5],
                     [21.5, 23.5]]]]]
    assert np.allclose(result, result_ref)
