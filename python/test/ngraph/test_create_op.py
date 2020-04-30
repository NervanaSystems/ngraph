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


integral_np_types = [np.int8, np.int16, np.int32, np.int64,
                     np.uint8, np.uint16, np.uint32, np.uint64]


@pytest.mark.parametrize('dtype', integral_np_types)
def test_interpolate(dtype):
    image_shape = [1, 3, 1024, 1024]
    output_shape = [64, 64]
    attributes = {
        'axes': [2, 3],
        'mode': 'cubic',
    }

    image_node = ng.parameter(image_shape, dtype, name='Image')

    node = ng.interpolate(image_node, output_shape, attributes)
    expected_shape = [1, 3, 1024, 1024]

    assert node.get_type_name() == 'Interpolate'
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


@pytest.mark.parametrize('int_dtype, fp_dtype', [
    (np.int8, np.float32),
    (np.int16, np.float32),
    (np.int32, np.float32),
    (np.int64, np.float32),
    (np.uint8, np.float32),
    (np.uint16, np.float32),
    (np.uint32, np.float32),
    (np.uint64, np.float32),
    (np.int32, np.float16),
    (np.int32, np.float64),
])
def test_prior_box(int_dtype, fp_dtype):
    image_shape = np.array([300, 300], dtype=int_dtype)
    attributes = {
        'offset': fp_dtype(0),
        'min_size': np.array([2, 3], dtype=fp_dtype),
        'aspect_ratio': np.array([1.5, 2.0, 2.5], dtype=fp_dtype),
    }

    layer_shape = ng.constant(np.array([32, 32], dtype=int_dtype), int_dtype)

    node = ng.prior_box(layer_shape, image_shape, attributes)

    assert node.get_type_name() == 'PriorBox'
    assert node.get_output_size() == 1
    assert node.get_output_shape(0) == [2, 20480]


@pytest.mark.parametrize('int_dtype, fp_dtype', [
    (np.int8, np.float32),
    (np.int16, np.float32),
    (np.int32, np.float32),
    (np.int64, np.float32),
    (np.uint8, np.float32),
    (np.uint16, np.float32),
    (np.uint32, np.float32),
    (np.uint64, np.float32),
    (np.int32, np.float16),
    (np.int32, np.float64),
])
def test_prior_box_clustered(int_dtype, fp_dtype):
    image_size = np.array([300, 300], dtype=int_dtype)
    attributes = {
        'offset': fp_dtype(0.5),
        'widths': np.array([4.0, 2.0, 3.2], dtype=fp_dtype),
        'heights': np.array([1.0, 2.0, 1.0], dtype=fp_dtype),
    }

    output_size = ng.constant(np.array([19, 19], dtype=int_dtype), int_dtype)

    node = ng.prior_box_clustered(output_size, image_size, attributes)

    assert node.get_type_name() == 'PriorBoxClustered'
    assert node.get_output_size() == 1
    assert node.get_output_shape(0) == [2, 4332]
