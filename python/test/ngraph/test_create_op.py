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
from ngraph.impl import Type
import test

np_types = [np.float32, np.int32]


@pytest.mark.parametrize('dtype', np_types)
def test_binary_convolution(dtype):
    strides = np.array([1, 1])
    pads_begin = np.array([0, 0])
    pads_end = np.array([0, 0])
    dilations = np.array([1, 1])
    mode = 'xnor-popcount'
    pad_value = 0.

    input0_shape = [1, 1, 9, 9]
    input1_shape = [1, 1, 3, 3]
    expected_shape = [1, 1, 7, 7]

    parameter_input0 = ng.parameter(input0_shape, name='Input0', dtype=dtype)
    parameter_input1 = ng.parameter(input1_shape, name='Input1', dtype=dtype)

    node = ng.binary_convolution(parameter_input0, parameter_input1,
                                 strides, pads_begin, pads_end, dilations, mode, pad_value)

    assert node.get_type_name() == 'BinaryConvolution'
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


@pytest.mark.parametrize('dtype', np_types)
def test_ctc_greedy_decoder(dtype):
    input0_shape = [20, 8, 128]
    input1_shape = [20, 8]
    expected_shape = [8, 20, 1, 1]

    parameter_input0 = ng.parameter(input0_shape, name='Input0', dtype=dtype)
    parameter_input1 = ng.parameter(input1_shape, name='Input1', dtype=dtype)

    node = ng.ctc_greedy_decoder(parameter_input0, parameter_input1)

    assert node.get_type_name() == 'CTCGreedyDecoder'
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


@pytest.mark.parametrize('dtype', np_types)
def test_deformable_convolution(dtype):
    strides = np.array([1, 1])
    pads_begin = np.array([0, 0])
    pads_end = np.array([0, 0])
    dilations = np.array([1, 1])

    input0_shape = [1, 1, 9, 9]
    input1_shape = [1, 1, 9, 9]
    input2_shape = [1, 1, 3, 3]
    expected_shape = [1, 1, 7, 7]

    parameter_input0 = ng.parameter(input0_shape, name='Input0', dtype=dtype)
    parameter_input1 = ng.parameter(input1_shape, name='Input1', dtype=dtype)
    parameter_input2 = ng.parameter(input2_shape, name='Input2', dtype=dtype)

    node = ng.deformable_convolution(parameter_input0, parameter_input1, parameter_input2,
                                     strides, pads_begin, pads_end, dilations)

    assert node.get_type_name() == 'DeformableConvolution'
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


@pytest.mark.parametrize('dtype', np_types)
def test_deformable_psroi_pooling(dtype):
    output_dim = 8
    spatial_scale = 0.0625
    group_size = 7
    mode = 'bilinear_deformable'
    spatial_bins_x = 4
    spatial_bins_y = 4
    trans_std = 0.1
    part_size = 7

    input0_shape = [1, 392, 38, 63]
    input1_shape = [300, 5]
    input2_shape = [300, 2, 7, 7]
    expected_shape = [300, 8, 7, 7]

    parameter_input0 = ng.parameter(input0_shape, name='Input0', dtype=dtype)
    parameter_input1 = ng.parameter(input1_shape, name='Input1', dtype=dtype)
    parameter_input2 = ng.parameter(input2_shape, name='Input2', dtype=dtype)

    node = ng.deformable_psroi_pooling(
        parameter_input0,
        parameter_input1,
        output_dim,
        spatial_scale,
        group_size,
        mode,
        spatial_bins_x,
        spatial_bins_y,
        trans_std,
        part_size,
        offsets=parameter_input2)

    assert node.get_type_name() == 'DeformablePSROIPooling'
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


@pytest.mark.parametrize('dtype', np_types)
def test_floor_mod(dtype):
    input0_shape = [8, 1, 6, 1]
    input1_shape = [7, 1, 5]
    expected_shape = [8, 7, 6, 5]

    parameter_input0 = ng.parameter(input0_shape, name='Input0', dtype=dtype)
    parameter_input1 = ng.parameter(input1_shape, name='Input1', dtype=dtype)

    node = ng.floor_mod(parameter_input0, parameter_input1)

    assert node.get_type_name() == 'FloorMod'
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


@pytest.mark.parametrize('dtype', np_types)
def test_gather_tree(dtype):
    input0_shape = [100, 1, 10]
    input1_shape = [100, 1, 10]
    input2_shape = [1]
    input3_shape = []
    expected_shape = [100, 1, 10]

    parameter_input0 = ng.parameter(input0_shape, name='Input0', dtype=dtype)
    parameter_input1 = ng.parameter(input1_shape, name='Input1', dtype=dtype)
    parameter_input2 = ng.parameter(input2_shape, name='Input2', dtype=dtype)
    parameter_input3 = ng.parameter(input3_shape, name='Input3', dtype=dtype)

    node = ng.gather_tree(parameter_input0, parameter_input1, parameter_input2, parameter_input3)

    assert node.get_type_name() == 'GatherTree'
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape


def test_roi_pooling():
    inputs = ng.parameter([2, 3, 4, 5], dtype=np.float32)
    coords = ng.parameter([150, 5], dtype=np.float32)
    node = ng.roi_pooling(inputs, coords, [6, 6], 0.0625, 'Max')

    assert node.get_type_name() == 'ROIPooling'
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [150, 3, 6, 6]
    assert node.get_output_element_type(0) == Type.f32


def test_psroi_pooling():
    inputs = ng.parameter([1, 3, 4, 5], dtype=np.float32)
    coords = ng.parameter([150, 5], dtype=np.float32)
    node = ng.psroi_pooling(inputs, coords, 2, 6, 0.0625, 0, 0, 'Avg')

    assert node.get_type_name() == 'PSROIPooling'
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [150, 2, 6, 6]
    assert node.get_output_element_type(0) == Type.f32


def test_convert_like():
    parameter_data = ng.parameter([1, 2, 3, 4], name='data', dtype=np.float32)
    like = ng.constant(1, dtype=np.int8)

    node = ng.convert_like(parameter_data, like)

    assert node.get_type_name() == 'ConvertLike'
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [1, 2, 3, 4]
    assert node.get_output_element_type(0) == Type.i8


def test_one_hot():
    data = np.array([0, 1, 2], dtype=np.int32)
    depth = 2
    on_value = 5
    off_value = 10
    axis = -1
    excepted = [[5, 10], [10, 5], [10, 10]]

    result = test.ngraph.util.run_op_node([data, depth, on_value, off_value], ng.ops.one_hot, axis)
    assert np.allclose(result, excepted)


def test_reverse():
    parameter_data = ng.parameter([3, 10, 100, 200], name='data', dtype=np.float32)
    parameter_axis = ng.parameter([1], name='axis', dtype=np.int64)
    expected_shape = [3, 10, 100, 200]

    node = ng.reverse(parameter_data, parameter_axis, 'index')

    assert node.get_type_name() == 'Reverse'
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == expected_shape
    assert node.get_output_element_type(0) == Type.f32


def test_select():
    cond = [[False, False], [True, False], [True, True]]
    then_node = [[-1, 0], [1, 2], [3, 4]]
    else_node = [[11, 10], [9, 8], [7, 6]]
    excepted = [[11, 10], [1, 8], [3, 4]]

    result = test.ngraph.util.run_op_node([cond, then_node, else_node], ng.ops.select)
    assert np.allclose(result, excepted)


def test_bucketize():
    data = ng.parameter([4, 3, 2, 1], name='data', dtype=np.float32)
    buckets = ng.parameter([5], name='buckets', dtype=np.int64)

    node = ng.bucketize(data, buckets, 'i32')

    assert node.get_type_name() == 'Bucketize'
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [4, 3, 2, 1]
    assert node.get_output_element_type(0) == Type.i32


def test_range():
    start = 5
    stop = 35
    step = 5

    result = test.ngraph.util.run_op_node([start, stop, step], ng.ops.range)
    assert np.allclose(result, [5, 10, 15, 20, 25, 30])


def test_region_yolo():
    data = ng.parameter([1, 125, 13, 13], name='input', dtype=np.float32)
    num_coords = 4
    num_classes = 80
    num_regions = 1
    mask = [6, 7, 8]
    axis = 0
    end_axis = 3
    do_softmax = False

    node = ng.region_yolo(data, num_coords, num_classes, num_regions,
                          mask, axis, end_axis, do_softmax)

    assert node.get_type_name() == 'RegionYolo'
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [1, (80 + 4 + 1) * 3, 13, 13]
    assert node.get_output_element_type(0) == Type.f32


def test_reorg_yolo():
    data = ng.parameter([2, 24, 34, 62], name='input', dtype=np.int32)
    stride = [2]

    node = ng.reorg_yolo(data, stride)

    assert node.get_type_name() == 'ReorgYolo'
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [2, 96, 17, 31]
    assert node.get_output_element_type(0) == Type.i32
