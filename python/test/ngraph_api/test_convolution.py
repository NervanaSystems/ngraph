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

import ngraph_api as ng


def test_default_arguments_convolution_2d():
    manager_name = pytest.config.getoption('backend', default='CPU')
    runtime = ng.runtime(manager_name=manager_name)
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

    assert np.array_equal(result,
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
    assert np.array_equal(result,
                          np.array([[[[-20, -20, 20, 20, 0, 0, 0],
                                      [-20, -20, 20, 20, 0, 0, 0],
                                      [-20, -20, 20, 20, 0, 0, 0],
                                      [-20, -20, 20, 20, 0, 0, 0],
                                      [-20, -20, 20, 20, 0, 0, 0],
                                      [-20, -20, 20, 20, 0, 0, 0],
                                      [-20, -20, 20, 20, 0, 0, 0]]]],
                                   dtype=np.float32))

    # convolution with strides=2 should produce 4 x 4 output:
    model = runtime.computation(ng.convolution(input_x, input_filter, strides=[2, 2]))
    result = model()
    assert np.array_equal(result,
                          np.array([[[[-20., 20., 0., 0.],
                                      [-20., 20., 0., 0.],
                                      [-20., 20., 0., 0.],
                                      [-20., 20., 0., 0.]]]],
                                   dtype=np.float32))

    # convolution with dilation=2 should produce 5 x 5 output:
    model = runtime.computation(ng.convolution(input_x, input_filter, dilation=(2, 2)))
    result = model()
    assert np.array_equal(result,
                          np.array([[[[0, 0, 20, 20, 0],
                                      [0, 0, 20, 20, 0],
                                      [0, 0, 20, 20, 0],
                                      [0, 0, 20, 20, 0],
                                      [0, 0, 20, 20, 0]]]],
                                   dtype=np.float32))
