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
    comp_topk = ng.topk(input_x, K, 0, 'max', 'SORT_VALUES')

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


@pytest.mark.skip_on_gpu
def test_non_max_suppression():

    boxes_shape = [1, 1000, 4]
    scores_shape = [1, 1, 1000]
    expected_shape = [0, 3]
    boxes_parameter = ng.parameter(boxes_shape, name='Boxes', dtype=np.float32)
    scores_parameter = ng.parameter(scores_shape, name='Scores', dtype=np.float32)

    node = ng.non_max_suppression(boxes_parameter, scores_parameter)

    assert node.get_type_name() == 'NonMaxSuppression'
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


@pytest.mark.skip_on_gpu
def test_non_zero():

    data_shape = [3, 10, 100, 200]

    data_parameter = ng.parameter(data_shape, name='Data', dtype=np.float32)

    node = ng.non_zero(data_parameter)

    assert node.get_type_name() == 'NonZero'
    assert node.get_output_size() == 1


@pytest.mark.skip_on_gpu
def test_roi_align():

    data_shape = [7, 256, 200, 200]
    rois = [1000, 4]
    batch_indices = [1000]
    expected_shape = [1000, 256, 6, 6]

    data_parameter = ng.parameter(data_shape, name='Data', dtype=np.float32)
    rois_parameter = ng.parameter(rois, name='Rois', dtype=np.float32)
    batch_indices_parameter = ng.parameter(batch_indices, name='Batch_indices', dtype=np.int32)
    pooled_h = 6
    pooled_w = 6
    sampling_ratio = 2
    spatial_scale = np.float32(16)
    mode = 'avg'

    node = ng.roi_align(data_parameter, rois_parameter, batch_indices_parameter,
                        pooled_h, pooled_w, sampling_ratio, spatial_scale, mode)

    assert node.get_type_name() == 'ROIAlign'
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape
