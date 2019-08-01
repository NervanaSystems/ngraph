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


def test_elu_operator():
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


def test_fake_quantize():
    runtime = get_runtime()

    data_value = np.arange(24.0, dtype=np.float32).reshape(1, 2, 3, 4)
    input_low_value = np.float32(0)
    input_high_value = np.float32(23)
    output_low_value = np.float32(2)
    output_high_value = np.float32(16)
    levels = np.float32(4)

    data_shape = [1, 2, 3, 4]
    bound_shape = []
    parameter_data = ng.parameter(data_shape, name='data', dtype=np.float32)
    parameter_input_low = ng.parameter(bound_shape, name='input_low', dtype=np.float32)
    parameter_input_high = ng.parameter(bound_shape, name='input_high', dtype=np.float32)
    parameter_output_low = ng.parameter(bound_shape, name='output_low', dtype=np.float32)
    parameter_output_high = ng.parameter(bound_shape, name='output_high', dtype=np.float32)

    model = ng.fake_quantize(parameter_data,
                             parameter_input_low,
                             parameter_input_high,
                             parameter_output_low,
                             parameter_output_high,
                             levels)
    computation = runtime.computation(model,
                                      parameter_data,
                                      parameter_input_low,
                                      parameter_input_high,
                                      parameter_output_low,
                                      parameter_output_high)

    result = computation(data_value,
                         input_low_value,
                         input_high_value,
                         output_low_value,
                         output_high_value)

    expected = np.array([[[[[2., 2., 2., 2.],
                            [6.6666669, 6.6666669, 6.6666669, 6.6666669],
                            [6.6666669, 6.6666669, 6.6666669, 6.6666669]],
                        [[11.33333301, 11.33333301, 11.33333301, 11.33333301],
                            [11.33333301, 11.33333301, 11.33333301, 11.33333301],
                            [16., 16., 16., 16.]]]]], dtype=np.float32)
    assert np.allclose(result, expected)
