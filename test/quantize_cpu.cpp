//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <string>

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/experimental/quantized_avg_pool.hpp"
#include "ngraph/op/experimental/quantized_conv.hpp"
#include "ngraph/op/experimental/quantized_conv_bias.hpp"
#include "ngraph/op/experimental/quantized_conv_relu.hpp"
#include "ngraph/op/experimental/quantized_max_pool.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

TEST(quantize_cpu, quantize_max_pool_2d_unsigned)
{
    vector<uint8_t> a_data = {0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0, 1};
    Shape shape_a{1, 1, 3, 5};
    Shape window_shape{2, 3};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{0, 0};
    Shape padding_above{0, 0};
    auto A = make_shared<op::Parameter>(element::u8, shape_a);
    Shape shape_r{1, 1, 2, 3};
    auto QMP = make_shared<op::QuantizedMaxPool>(
        A, window_shape, window_movement_strides, padding_below, padding_above);
    auto f = make_shared<Function>(NodeVector{QMP}, op::ParameterVector{A});
    auto backend = runtime::Backend::create("CPU");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::u8, shape_a);
    copy_data(a, a_data);
    auto result = backend->create_tensor(element::u8, shape_r);
    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((vector<uint8_t>{3, 3, 2, 3, 3, 2}), read_vector<uint8_t>(result));
}

TEST(quantize_cpu, quantize_max_pool_2d_signed)
{
    vector<int8_t> a_data = {0, 1, 0, -2, 1, 0, -3, 2, 0, 0, 2, 0, 0, 0, 1};
    Shape shape_a{1, 1, 3, 5};
    Shape window_shape{2, 3};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{0, 0};
    Shape padding_above{0, 0};
    auto A = make_shared<op::Parameter>(element::i8, shape_a);
    Shape shape_r{1, 1, 2, 3};
    auto QMP = make_shared<op::QuantizedMaxPool>(
        A, window_shape, window_movement_strides, padding_below, padding_above);
    auto f = make_shared<Function>(NodeVector{QMP}, op::ParameterVector{A});
    auto backend = runtime::Backend::create("CPU");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i8, shape_a);
    copy_data(a, a_data);
    auto result = backend->create_tensor(element::i8, shape_r);
    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((vector<int8_t>{2, 2, 2, 2, 2, 2}), read_vector<int8_t>(result));
}

TEST(quantize_cpu, quantize_avg_pool_2d_unsigned)
{
    vector<uint8_t> a_data = {0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0, 1};
    Shape shape_a{1, 1, 3, 5};
    Shape window_shape{2, 3};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{0, 0};
    Shape padding_above{0, 0};
    auto A = make_shared<op::Parameter>(element::u8, shape_a);
    Shape shape_r{1, 1, 2, 3};
    auto QMP = make_shared<op::QuantizedAvgPool>(
        A, window_shape, window_movement_strides, padding_below, padding_above, false);
    auto f = make_shared<Function>(NodeVector{QMP}, op::ParameterVector{A});
    auto backend = runtime::Backend::create("CPU");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::u8, shape_a);
    copy_data(a, a_data);
    auto result = backend->create_tensor(element::u8, shape_r);
    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((vector<uint8_t>{1, 1, 1, 1, 1, 0}), read_vector<uint8_t>(result));
}

TEST(quantize_cpu, quantize_avg_pool_2d_signed)
{
    vector<int8_t> a_data = {10, 1, 0, -2, 1, 0, -3, 4, 0, 0, 2, 0, 0, 0, 1};
    Shape shape_a{1, 1, 3, 5};
    Shape window_shape{2, 3};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{0, 0};
    Shape padding_above{0, 0};
    auto A = make_shared<op::Parameter>(element::i8, shape_a);
    Shape shape_r{1, 1, 2, 3};
    auto QMP = make_shared<op::QuantizedAvgPool>(
        A, window_shape, window_movement_strides, padding_below, padding_above, false);
    auto f = make_shared<Function>(NodeVector{QMP}, op::ParameterVector{A});
    auto backend = runtime::Backend::create("CPU");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i8, shape_a);
    copy_data(a, a_data);
    auto result = backend->create_tensor(element::i8, shape_r);
    backend->call_with_validate(f, {result}, {a});
    EXPECT_EQ((vector<int8_t>{2, 0, 0, 0, 0, 1}), read_vector<int8_t>(result));
}

TEST(quantize_cpu, quantizedConv2D_small)
{
    Shape shape_a{1, 1, 3, 4}; // input shape
    Shape shape_b{1, 1, 3, 3}; // filter shape
    Shape shape_r{1, 1, 3, 4}; // output shape
    vector<uint8_t> a_data = {1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4};
    vector<int8_t> b_data = {1, 2, 3, 4, 5, 0, 0, 1, 2};
    auto A = make_shared<op::Parameter>(element::u8, shape_a);
    auto B = make_shared<op::Parameter>(element::i8, shape_b);
    auto C = op::Constant::create(element::f32, Shape{1}, {1.41664f});
    auto CV = make_shared<op::QuantizedConvolution>(A,
                                                    B,
                                                    Strides{1, 1},        // move_strides
                                                    Strides{1, 1},        // filter_dilation
                                                    CoordinateDiff{1, 1}, // below_pads
                                                    CoordinateDiff{1, 1}, // above_pads
                                                    Strides{1, 1},        // data_dilation
                                                    C);
    auto f = make_shared<Function>(NodeVector{CV}, op::ParameterVector{A, B});
    auto backend = runtime::Backend::create("CPU");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::u8, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::i8, shape_b);
    copy_data(b, b_data);
    auto result = backend->create_tensor(element::i8, shape_r);
    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<int8_t>{31, 48, 42, 45, 54, 102, 127, 61, 47, 74, 61, 55}),
              read_vector<int8_t>(result));
}

TEST(quantize_cpu, quantizedConv2D_with_relu)
{
    Shape shape_a{1, 1, 3, 4}; // input shape
    Shape shape_b{1, 1, 3, 3}; // filter shape
    Shape shape_r{1, 1, 3, 4}; // output shape
    vector<uint8_t> a_data = {1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4};
    vector<int8_t> b_data = {1, 2, 3, 4, 5, 0, 0, 1, 2};
    auto A = make_shared<op::Parameter>(element::u8, shape_a);
    auto B = make_shared<op::Parameter>(element::i8, shape_b);
    auto C = op::Constant::create(element::f32, Shape{1}, {1.41664f});
    auto CV = make_shared<op::QuantizedConvolutionRelu>(A,
                                                        B,
                                                        Strides{1, 1},        // move_strides
                                                        Strides{1, 1},        // filter_dilation
                                                        CoordinateDiff{1, 1}, // below_pads
                                                        CoordinateDiff{1, 1}, // above_pads
                                                        Strides{1, 1},        // data_dilation
                                                        C);
    auto f = make_shared<Function>(NodeVector{CV}, op::ParameterVector{A, B});
    auto backend = runtime::Backend::create("CPU");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::u8, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::i8, shape_b);
    copy_data(b, b_data);
    auto result = backend->create_tensor(element::u8, shape_r);
    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<uint8_t>{31, 48, 42, 45, 54, 102, 127, 61, 47, 74, 61, 55}),
              read_vector<uint8_t>(result));
}

TEST(quantize_cpu, quantizedConv2D_fused_relu)
{
    Shape shape_a{1, 1, 3, 3}; // input shape
    Shape shape_b{1, 1, 3, 3}; // filter shape
    Shape shape_r{1, 1, 3, 3}; // output shape
    vector<uint8_t> a_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    vector<int8_t> b_data = {1, 2, 1, 0, 0, 0, -1, -2, -1};
    auto A = make_shared<op::Parameter>(element::u8, shape_a);
    auto B = make_shared<op::Parameter>(element::i8, shape_b);
    auto C = op::Constant::create(element::f32, Shape{1}, {5.31242f});
    auto CV = make_shared<op::QuantizedConvolutionRelu>(A,
                                                        B,
                                                        Strides{1, 1},        // move_strides
                                                        Strides{1, 1},        // filter_dilation
                                                        CoordinateDiff{1, 1}, // below_pads
                                                        CoordinateDiff{1, 1}, // above_pads
                                                        Strides{1, 1},        // data_dilation
                                                        C);
    auto f = make_shared<Function>(NodeVector{CV}, op::ParameterVector{A, B});
    auto backend = runtime::Backend::create("CPU");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::u8, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::i8, shape_b);
    copy_data(b, b_data);
    auto result = backend->create_tensor(element::u8, shape_r);
    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<uint8_t>{0, 0, 0, 0, 0, 0, 69, 106, 90}), read_vector<uint8_t>(result));
}

TEST(quantize_cpu, quantizedConv2D_with_bias)
{
    Shape shape_a{1, 1, 3, 4}; // input shape
    Shape shape_b{1, 1, 3, 3}; // filter shape
    Shape shape_r{1, 1, 3, 4}; // output shape
    vector<uint8_t> a_data = {1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4};
    vector<int8_t> b_data = {1, 2, 3, 4, 5, 0, 0, 1, 2};
    vector<int32_t> c_data = {5};
    auto A = make_shared<op::Parameter>(element::u8, shape_a);
    auto B = make_shared<op::Parameter>(element::i8, shape_b);
    auto Bias = make_shared<op::Parameter>(element::i32, Shape{1});
    auto C = op::Constant::create(element::f32, Shape{1}, {1.41664f});
    auto CV = make_shared<op::QuantizedConvolutionBias>(A,
                                                        B,
                                                        Bias,
                                                        Strides{1, 1},        // move_strides
                                                        Strides{1, 1},        // filter_dilation
                                                        CoordinateDiff{1, 1}, // below_pads
                                                        CoordinateDiff{1, 1}, // above_pads
                                                        Strides{1, 1},        // data_dilation
                                                        C);
    auto f = make_shared<Function>(NodeVector{CV}, op::ParameterVector{A, B, Bias});
    auto backend = runtime::Backend::create("CPU");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::u8, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::i8, shape_b);
    copy_data(b, b_data);
    auto c = backend->create_tensor(element::i32, Shape{1});
    copy_data(c, c_data);
    auto result = backend->create_tensor(element::i8, shape_r);
    backend->call_with_validate(f, {result}, {a, b, c});
    EXPECT_EQ((vector<int8_t>{38, 55, 50, 52, 61, 109, 127, 68, 54, 81, 68, 62}),
              read_vector<int8_t>(result));
}

TEST(quantize_cpu, quantizedConv2D_with_bias_and_relu)
{
    Shape shape_a{1, 1, 3, 3}; // input shape
    Shape shape_b{1, 1, 3, 3}; // filter shape
    Shape shape_r{1, 1, 3, 3}; // output shape
    vector<uint8_t> a_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    vector<int8_t> b_data = {1, 2, 1, 0, 0, 0, -1, -2, -1};
    vector<int32_t> c_data = {5};
    auto A = make_shared<op::Parameter>(element::u8, shape_a);
    auto B = make_shared<op::Parameter>(element::i8, shape_b);
    auto Bias = make_shared<op::Parameter>(element::i32, Shape{1});
    auto C = op::Constant::create(element::f32, Shape{1}, {5.31242f});
    auto CV = make_shared<op::QuantizedConvolutionBias>(A,
                                                        B,
                                                        Bias,
                                                        Strides{1, 1},        // move_strides
                                                        Strides{1, 1},        // filter_dilation
                                                        CoordinateDiff{1, 1}, // below_pads
                                                        CoordinateDiff{1, 1}, // above_pads
                                                        Strides{1, 1},        // data_dilation
                                                        C,
                                                        true);
    auto f = make_shared<Function>(NodeVector{CV}, op::ParameterVector{A, B, Bias});
    auto backend = runtime::Backend::create("CPU");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::u8, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::i8, shape_b);
    copy_data(b, b_data);
    auto c = backend->create_tensor(element::i32, Shape{1});
    copy_data(c, c_data);
    auto result = backend->create_tensor(element::u8, shape_r);
    backend->call_with_validate(f, {result}, {a, b, c});
    EXPECT_EQ((vector<uint8_t>{0, 0, 0, 0, 0, 0, 96, 133, 117}), read_vector<uint8_t>(result));
}
