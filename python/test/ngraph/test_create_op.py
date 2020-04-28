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


np_types = [np.float32, np.float64, np.int8, np.int16, np.int32, np.int64,
            np.uint8, np.uint16, np.uint32, np.uint64]


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
