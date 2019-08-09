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


def test_depth_to_space():
    runtime = get_runtime()

    data_value = np.array([[[[0, 1, 2],
                            [3, 4, 5]],
                            [[6, 7, 8],
                            [9, 10, 11]],
                            [[12, 13, 14],
                            [15, 16, 17]],
                            [[18, 19, 20],
                            [21, 22, 23]]]], dtype=np.float32)
    block_size = np.float32(2)

    data_shape = [1, 4, 2, 3]
    parameter_data = ng.parameter(data_shape, name='Data', dtype=np.float32)

    model = ng.depth_to_space(parameter_data, block_size)
    computation = runtime.computation(model, parameter_data)

    result = computation(data_value)
    expected = np.array([[[[0, 6, 1, 7, 2, 8],
                        [12, 18, 13, 19, 14, 20],
                        [3, 9, 4, 10, 5, 11],
                        [15, 21, 16, 22, 17, 23]]]], dtype=np.float32)
    assert np.allclose(result, expected)


def test_gemm_operator():
    runtime = get_runtime()

    shape_a = [3, 2]
    shape_b = [3, 2]
    shape_c = [2, 1]

    value_a = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    value_b = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    value_c = np.array([[13], [14]], dtype=np.float32)

    parameter_a = ng.parameter(shape_a, name='A', dtype=np.float32)
    parameter_b = ng.parameter(shape_b, name='B', dtype=np.float32)
    parameter_c = ng.parameter(shape_c, name='C', dtype=np.float32)

    alpha_value = np.float32(3)
    beta_value = np.float32(3)

    transA = True
    transB = False

    model = ng.gemm(parameter_a, parameter_b, parameter_c, alpha_value, beta_value, transA, transB)
    computation = runtime.computation(model, parameter_a, parameter_b, parameter_c)

    result = computation(value_a, value_b, value_c)

    # expected = value_alpha * value_a' * value_b + value_beta * value_c

    value_a = value_a.transpose()
    a_mul_a = np.multiply(alpha_value, value_a)
    aa_mul_b = np.dot(a_mul_a, value_b)
    b_mul_c = np.dot(beta_value, value_c)
    expected = np.add(aa_mul_b, b_mul_c)

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


def test_unsqueeze():
    runtime = get_runtime()

    data_shape = [3, 4, 5]
    parameter_data = ng.parameter(data_shape, name='Data', dtype=np.float32)
    data_value = np.arange(60., dtype=np.float32).reshape(3, 4, 5)
    axes = [0, 4]
    model = ng.unsqueeze(parameter_data, axes)
    computation = runtime.computation(model, parameter_data)

    result = computation(data_value)
    expected = np.arange(60., dtype=np.float32).reshape(1, 3, 4, 5, 1)
    assert np.allclose(result, expected)


def test_grn_operator():
    runtime = get_runtime()

    data_value = np.arange(start=1.0, stop=25.0, dtype=np.float32).reshape(1, 2, 3, 4)
    bias = np.float32(1e-6)

    data_shape = [1, 2, 3, 4]

    parameter_data = ng.parameter(data_shape, name='Data', dtype=np.float32)

    model = ng.grn(parameter_data, bias)
    computation = runtime.computation(model, parameter_data)

    result = computation(data_value)
    expected = np.array([[[[0.0766965, 0.14142136, 0.19611613, 0.24253564],
                           [0.28216633, 0.31622776, 0.34570536, 0.37139067],
                           [0.39391932, 0.41380295, 0.4314555, 0.4472136]],
                          [[0.9970545, 0.98994946, 0.9805807, 0.97014254],
                           [0.9593655, 0.9486833, 0.9383431, 0.9284767],
                           [0.91914505, 0.9103665, 0.9021342, 0.8944272]]]], dtype=np.float32)

    assert np.allclose(result, expected)
