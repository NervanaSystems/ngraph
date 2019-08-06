# ******************************************************************************
# Copyright 2017-2019 Intel Corporation
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


def test_elu_operator_with_parameters():
    runtime = get_runtime()

    data_shape = [2, 2]
    alpha_shape = [2]
    parameter_data = ng.parameter(data_shape, name='Data', dtype=np.float32)
    parameter_alpha = ng.parameter(alpha_shape, name='Alpha', dtype=np.float32)

    model = ng.elu(parameter_data, parameter_alpha)
    computation = runtime.computation(model, parameter_data, parameter_alpha)

    value_data = np.array([[-5, 1], [-2, 3]], dtype=np.float32)
    value_alpha = np.array([3, 3], dtype=np.float32)

    result = computation(value_data, value_alpha)
    expected = np.array([[-2.9797862, 1.], [-2.5939941, 3.]], dtype=np.float32)
    assert np.allclose(result, expected)


def test_elu_operator_with_scalar_and_array():
    runtime = get_runtime()

    data_value = np.array([[-5, 1], [-2, 3]], dtype=np.float32)
    alpha_value = np.float32(3)

    model = ng.elu(data_value, alpha_value)
    computation = runtime.computation(model)

    result = computation()
    expected = np.array([[-2.9797862, 1.], [-2.5939941, 3.]], dtype=np.float32)
    assert np.allclose(result, expected)


def test_elu_operator_with_scalar():
    runtime = get_runtime()

    data_value = np.array([[-5, 1], [-2, 3]], dtype=np.float32)
    alpha_value = np.float32(3)

    data_shape = [2, 2]
    parameter_data = ng.parameter(data_shape, name='Data', dtype=np.float32)

    model = ng.elu(parameter_data, alpha_value)
    computation = runtime.computation(model, parameter_data)

    result = computation(data_value)
    expected = np.array([[-2.9797862, 1.], [-2.5939941, 3.]], dtype=np.float32)
    assert np.allclose(result, expected)


def test_gelu_operator_with_parameters():
    runtime = get_runtime()

    data_value = np.array([[-5, 1], [-2, 3]], dtype=np.float32)

    data_shape = [2, 2]
    parameter_data = ng.parameter(data_shape, name='Data', dtype=np.float32)

    model = ng.gelu(parameter_data)
    computation = runtime.computation(model, parameter_data)

    result = computation(data_value)
    expected = np.array([[-1.4901161e-06, 8.4134471e-01], [-4.5500278e-02, 2.9959502]],
                        dtype=np.float32)
    assert np.allclose(result, expected)


def test_gelu_operator_with_array():
    runtime = get_runtime()

    data_value = np.array([[-5, 1], [-2, 3]], dtype=np.float32)

    model = ng.gelu(data_value)
    computation = runtime.computation(model)

    result = computation()
    expected = np.array([[-1.4901161e-06, 8.4134471e-01], [-4.5500278e-02, 2.9959502]],
                        dtype=np.float32)

    assert np.allclose(result, expected)


def test_clamp_operator():
    runtime = get_runtime()

    data_shape = [2, 2]
    parameter_data = ng.parameter(data_shape, name='Data', dtype=np.float32)
    min_value = np.float32(3)
    max_value = np.float32(12)

    model = ng.clamp(parameter_data, min_value, max_value)
    computation = runtime.computation(model, parameter_data)

    data_value = np.array([[-5, 9], [45, 3]], dtype=np.float32)

    result = computation(data_value)
    expected = np.clip(data_value, min_value, max_value)
    assert np.allclose(result, expected)


def test_clamp_operator_with_array():
    runtime = get_runtime()

    data_value = np.array([[-5, 9], [45, 3]], dtype=np.float32)
    min_value = np.float32(3)
    max_value = np.float32(12)

    model = ng.clamp(data_value, min_value, max_value)
    computation = runtime.computation(model)

    result = computation()
    expected = np.clip(data_value, min_value, max_value)

    assert np.allclose(result, expected)


def test_squeeze_operator():
    runtime = get_runtime()

    data_shape = [1, 2, 1, 3, 1, 1]
    parameter_data = ng.parameter(data_shape, name='Data', dtype=np.float32)
    data_value = np.arange(6., dtype=np.float32).reshape(1, 2, 1, 3, 1, 1)
    axes = [2, 4]
    model = ng.squeeze(parameter_data, axes)
    computation = runtime.computation(model, parameter_data)

    result = computation(data_value)
    expected = np.arange(6., dtype=np.float32).reshape(1, 2, 3, 1)
    assert np.allclose(result, expected)


def test_squared_difference_operator():
    runtime = get_runtime()

    x1_shape = [1, 2, 3, 4]
    x2_shape = [2, 3, 4]

    parameter_x1 = ng.parameter(x1_shape, name='x1', dtype=np.float32)
    parameter_x2 = ng.parameter(x2_shape, name='x2', dtype=np.float32)

    x1_value = np.arange(24., dtype=np.float32).reshape(x1_shape)
    x2_value = np.arange(start=4., stop=28., step=1.0, dtype=np.float32).reshape(x2_shape)

    model = ng.squared_difference(parameter_x1, parameter_x2)
    computation = runtime.computation(model, parameter_x1, parameter_x2)

    result = computation(x1_value, x2_value)
    expected = np.square(np.subtract(x1_value, x2_value))
    assert np.allclose(result, expected)


def test_shuffle_channels_operator():
    runtime = get_runtime()

    data_shape = [1, 15, 2, 2]
    axis = 1
    groups = 5

    parameter = ng.parameter(data_shape, name='Data', dtype=np.float32)

    data_value = np.arange(60., dtype=np.float32).reshape(data_shape)

    model = ng.shuffle_channels(parameter, axis, groups)
    computation = runtime.computation(model, parameter)

    result = computation(data_value)
    expected = np.array([[[[0., 1.], [2., 3.]], [[12., 13.], [14., 15.]],
                          [[24., 25.], [26., 27.]], [[36., 37.], [38., 39.]],
                          [[48., 49.], [50., 51.]], [[4., 5.], [6., 7.]],
                          [[16., 17.], [18., 19.]], [[28., 29.], [30., 31.]],
                          [[40., 41.], [42., 43.]], [[52., 53.], [54., 55.]],
                          [[8., 9.], [10., 11.]], [[20., 21.], [22., 23.]],
                          [[32., 33.], [34., 35.]], [[44., 45.], [46., 47.]],
                          [[56., 57.], [58., 59.]]]], dtype=np.float32)
    assert np.allclose(result, expected)
