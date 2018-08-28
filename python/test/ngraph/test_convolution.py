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


@pytest.config.gpu_skip(reason='Not implemented')
def test_convolution_2d():
    runtime = get_runtime()
    # input_x should have shape N(batch) x C x H x W
    input_x = ng.constant(np.array([
        [0., 0., 5., 5., 0., 0., 0., 0., 0.],
        [0., 0., 5., 5., 0., 0., 0., 0., 0.],
        [0., 0., 5., 5., 0., 0., 0., 0., 0.],
        [0., 0., 5., 5., 0., 0., 0., 0., 0.],
        [0., 0., 5., 5., 0., 0., 0., 0., 0.],
        [0., 0., 5., 5., 0., 0., 0., 0., 0.],
        [0., 0., 5., 5., 0., 0., 0., 0., 0.],
        [0., 0., 5., 5., 0., 0., 0., 0., 0.],
        [0., 0., 5., 5., 0., 0., 0., 0., 0.]], dtype=np.float32).reshape(1, 1, 9, 9))

    # filter weights should have shape M x C x kH x kW
    input_filter = ng.constant(np.array([
        [1., 0., -1.],
        [2., 0., -2.],
        [1., 0., -1.]], dtype=np.float32).reshape(1, 1, 3, 3))

    # convolution with padding=1 should produce 9 x 9 output:
    model = runtime.computation(ng.convolution(input_x, input_filter,
                                               padding_above=[1, 1], padding_below=[1, 1]))
    result = model()

    assert np.allclose(result,
                       np.array([[[[0., -15., -15., 15., 15., 0., 0., 0., 0.],
                                   [0., -20., -20., 20., 20., 0., 0., 0., 0.],
                                   [0., -20., -20., 20., 20., 0., 0., 0., 0.],
                                   [0., -20., -20., 20., 20., 0., 0., 0., 0.],
                                   [0., -20., -20., 20., 20., 0., 0., 0., 0.],
                                   [0., -20., -20., 20., 20., 0., 0., 0., 0.],
                                   [0., -20., -20., 20., 20., 0., 0., 0., 0.],
                                   [0., -20., -20., 20., 20., 0., 0., 0., 0.],
                                   [0., -15., -15., 15., 15., 0., 0., 0., 0.]]]],
                                dtype=np.float32))

    # convolution with padding=0 should produce 7 x 7 output:
    model = runtime.computation(ng.convolution(input_x, input_filter))
    result = model()
    assert np.allclose(result,
                       np.array([[[[-20, -20, 20, 20, 0, 0, 0],
                                   [-20, -20, 20, 20, 0, 0, 0],
                                   [-20, -20, 20, 20, 0, 0, 0],
                                   [-20, -20, 20, 20, 0, 0, 0],
                                   [-20, -20, 20, 20, 0, 0, 0],
                                   [-20, -20, 20, 20, 0, 0, 0],
                                   [-20, -20, 20, 20, 0, 0, 0]]]],
                                dtype=np.float32))

    # convolution with strides=2 should produce 4 x 4 output:
    model = runtime.computation(ng.convolution(input_x, input_filter, filter_strides=[2, 2]))
    result = model()
    assert np.allclose(result,
                       np.array([[[[-20., 20., 0., 0.],
                                   [-20., 20., 0., 0.],
                                   [-20., 20., 0., 0.],
                                   [-20., 20., 0., 0.]]]],
                                dtype=np.float32))

    # convolution with dilation=2 should produce 5 x 5 output:
    model = runtime.computation(ng.convolution(input_x, input_filter,
                                               filter_dilation_strides=(2, 2)))
    result = model()
    assert np.allclose(result,
                       np.array([[[[0, 0, 20, 20, 0],
                                   [0, 0, 20, 20, 0],
                                   [0, 0, 20, 20, 0],
                                   [0, 0, 20, 20, 0],
                                   [0, 0, 20, 20, 0]]]],
                                dtype=np.float32))


@pytest.config.gpu_skip(reason='Not implemented')
def test_convolution_backprop_data():
    runtime = get_runtime()

    data_batch_shape = [1, 1, 9, 9]
    filter_shape = [1, 1, 3, 3]
    output_delta_shape = [1, 1, 7, 7]

    filter_param = ng.parameter(shape=filter_shape)
    output_delta_param = ng.parameter(shape=output_delta_shape)

    deconvolution = ng.convolution_backprop_data(data_batch_shape, filter_param, output_delta_param)

    data_batch_data = np.array([[[[-20, -20, 20, 20, 0, 0, 0],
                                  [-20, -20, 20, 20, 0, 0, 0],
                                  [-20, -20, 20, 20, 0, 0, 0],
                                  [-20, -20, 20, 20, 0, 0, 0],
                                  [-20, -20, 20, 20, 0, 0, 0],
                                  [-20, -20, 20, 20, 0, 0, 0],
                                  [-20, -20, 20, 20, 0, 0, 0]]]],
                               dtype=np.float32)

    filter_data = np.array([
        [1., 0., -1.],
        [2., 0., -2.],
        [1., 0., -1.]], dtype=np.float32).reshape(1, 1, 3, 3)

    model = runtime.computation(deconvolution, filter_param, output_delta_param)
    result = model(filter_data, data_batch_data)
    assert np.allclose(result,
                       np.array([[[[-20., -20., 40., 40., -20., -20., 0., 0., 0.],
                                   [-60., -60., 120., 120., -60., -60., 0., 0., 0.],
                                   [-80., -80., 160., 160., -80., -80., 0., 0., 0.],
                                   [-80., -80., 160., 160., -80., -80., 0., 0., 0.],
                                   [-80., -80., 160., 160., -80., -80., 0., 0., 0.],
                                   [-80., -80., 160., 160., -80., -80., 0., 0., 0.],
                                   [-80., -80., 160., 160., -80., -80., 0., 0., 0.],
                                   [-60., -60., 120., 120., -60., -60., 0., 0., 0.],
                                   [-20., -20., 40., 40., -20., -20., 0., 0., 0.]]]],
                                dtype=np.float32))
