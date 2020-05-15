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

import ngraph as ng
from test.ngraph.util import get_runtime
from ngraph.impl.op import Quantize


def test_quantize_operator():
    runtime = get_runtime()

    data_shape = [6]
    scale_shape = []
    zero_point_shape = []

    data_value = np.array([0, 2, 3, 1000, -254, -1000]).astype(np.float32)
    scale_value = np.float32(2)
    zero_point_value = np.uint8(128)
    new_type = np.uint8
    axis_set = []

    parameter_data = ng.parameter(data_shape, name='Data', dtype=np.float32)
    parameter_scale = ng.parameter(scale_shape, name='Scale', dtype=np.float32)
    parameter_zero_point = ng.parameter(zero_point_shape, name='Zero_Point', dtype=np.uint8)

    model = ng.quantize(parameter_data,
                        parameter_scale,
                        parameter_zero_point,
                        new_type,
                        axis_set,
                        Quantize.RoundMode.ROUND_NEAREST_TOWARD_INFINITY)
    computation = runtime.computation(model,
                                      parameter_data,
                                      parameter_scale,
                                      parameter_zero_point)

    result = computation(data_value, scale_value, zero_point_value)
    expected = np.array([128, 129, 130, 255, 1, 0]).astype(np.uint8)
    assert np.allclose(result, expected)


def test_quantized_convoluction_operator():
    runtime = get_runtime()

    data_shape = [1, 1, 3, 4]
    filters_shape = [1, 1, 3, 3]
    result_shape = [1, 1, 3, 4]
    shape = []

    data_value = np.array([1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4]).astype(np.uint8).reshape(data_shape)
    filters_value = np.array([1, 2, 3, 4, 5, 0, 0, 1, 2]).astype(np.uint8).reshape(filters_shape)
    window_movement_strides = [1, 1]
    window_dilation_strides = [1, 1]
    padding_below = [1, 1]
    padding_above = [1, 1]
    data_dilation_strides = [1, 1]
    input_scale_value = 1
    input_zero_point_value = 0
    filter_scale_value = 1
    filter_zero_point_value = 0
    output_scale_value = 1
    output_zero_point_value = 0
    output_type = np.int32
    input_axes = []
    filter_axes = []
    output_axes = []

    parameter_data = ng.parameter(data_shape, name='Data', dtype=np.uint8)
    parameter_filters = ng.parameter(filters_shape, name='Filters', dtype=np.uint8)
    parameter_input_scale = ng.parameter(shape, name='Input_scale', dtype=np.float32)
    parameter_input_zero_point = ng.parameter(shape, name='Input_zero_point', dtype=np.uint8)
    parameter_filter_scale = ng.parameter(shape, name='Filter_scale', dtype=np.float32)
    parameter_filter_zero_point = ng.parameter(shape, name='Filter_zero_point', dtype=np.uint8)
    parameter_output_scale = ng.parameter(shape, name='Output_scale', dtype=np.float32)
    parameter_output_zero_point = ng.parameter(shape, name='Output_zero_point', dtype=np.int32)

    model = ng.quantized_convolution(parameter_data,
                                     parameter_filters,
                                     window_movement_strides,
                                     window_dilation_strides,
                                     padding_below,
                                     padding_above,
                                     data_dilation_strides,
                                     parameter_input_scale,
                                     parameter_input_zero_point,
                                     parameter_filter_scale,
                                     parameter_filter_zero_point,
                                     parameter_output_scale,
                                     parameter_output_zero_point,
                                     output_type,
                                     input_axes,
                                     filter_axes,
                                     output_axes)
    computation = runtime.computation(model,
                                      parameter_data,
                                      parameter_filters,
                                      parameter_input_scale,
                                      parameter_input_zero_point,
                                      parameter_filter_scale,
                                      parameter_filter_zero_point,
                                      parameter_output_scale,
                                      parameter_output_zero_point)

    result = computation(data_value,
                         filters_value,
                         input_scale_value,
                         input_zero_point_value,
                         filter_scale_value,
                         filter_zero_point_value,
                         output_scale_value,
                         output_zero_point_value)
    expected = np.array([22, 34, 30, 32, 38, 72,
                         90, 43, 33, 52, 43, 39]).astype(np.int8).reshape(result_shape)
    assert np.allclose(result, expected)


def test_quantized_dot_operator():
    runtime = get_runtime()

    input0_shape = [1, 2]
    input1_shape = [2, 3]
    result_shape = [1, 3]
    shape = []

    input0_value = np.array([2, 3]).astype(np.uint8).reshape(input0_shape)
    input1_value = np.array([0, 2, 4, 1, 3, 5]).astype(np.uint8).reshape(input1_shape)
    reduction_axes_count = 1
    input0_scale_value = 2
    input0_zero_point_value = 0
    input1_scale_value = 1
    input1_zero_point_value = 0
    output_scale_value = 2
    output_zero_point_value = 0
    output_type = np.uint8
    input0_axes = []
    input1_axes = []
    output_axes = []

    parameter_input0 = ng.parameter(input0_shape, name='Input0', dtype=np.uint8)
    parameter_input1 = ng.parameter(input1_shape, name='Input1', dtype=np.uint8)
    parameter_input0_scale = ng.parameter(shape, name='Input0_scale', dtype=np.float32)
    parameter_input0_zero_point = ng.parameter(shape, name='Input0_zero_point', dtype=np.uint8)
    parameter_input1_scale = ng.parameter(shape, name='Input1_scale', dtype=np.float32)
    parameter_input1_zero_point = ng.parameter(shape, name='Input1_zero_point', dtype=np.uint8)
    parameter_output_scale = ng.parameter(shape, name='Output_scale', dtype=np.float32)
    parameter_output_zero_point = ng.parameter(shape, name='Output_zero_point', dtype=np.uint8)

    model = ng.quantized_dot(parameter_input0,
                             parameter_input1,
                             reduction_axes_count,
                             parameter_input0_scale,
                             parameter_input0_zero_point,
                             parameter_input1_scale,
                             parameter_input1_zero_point,
                             parameter_output_scale,
                             parameter_output_zero_point,
                             output_type,
                             input0_axes,
                             input1_axes,
                             output_axes)
    computation = runtime.computation(model,
                                      parameter_input0,
                                      parameter_input1,
                                      parameter_input0_scale,
                                      parameter_input0_zero_point,
                                      parameter_input1_scale,
                                      parameter_input1_zero_point,
                                      parameter_output_scale,
                                      parameter_output_zero_point)

    result = computation(input0_value,
                         input1_value,
                         input0_scale_value,
                         input0_zero_point_value,
                         input1_scale_value,
                         input1_zero_point_value,
                         output_scale_value,
                         output_zero_point_value)
    expected = np.array([3, 13, 23]).astype(np.int8).reshape(result_shape)
    assert np.allclose(result, expected)


def test_dequantize_operator():
    runtime = get_runtime()

    data_shape = [4, 3]
    scale_shape = []
    zero_point_shape = []
    result_shape = [4, 3]

    data_value = np.array([1, 1, 2, -1, 3, -1,
                           4, -3, 5, -3, 6, -5]).astype(np.int8).reshape(data_shape)
    scale_value = np.float32(2)
    zero_point_value = np.int8(1)
    element_type = np.float32
    axis_set = []

    parameter_data = ng.parameter(data_shape, name='Data', dtype=np.int8)
    parameter_scale = ng.parameter(scale_shape, name='Scale', dtype=np.float32)
    parameter_zero_point = ng.parameter(zero_point_shape, name='Zero_Point', dtype=np.int8)

    model = ng.dequantize(parameter_data,
                          parameter_scale,
                          parameter_zero_point,
                          element_type,
                          axis_set)
    computation = runtime.computation(model,
                                      parameter_data,
                                      parameter_scale,
                                      parameter_zero_point)

    result = computation(data_value, scale_value, zero_point_value)
    expected = np.array([0, 0, 2, -4, 4, -4,
                         6, -8, 8, -8, 10, -12]).astype(np.float32).reshape(result_shape)
    assert np.allclose(result, expected)
