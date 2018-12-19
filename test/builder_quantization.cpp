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
#include "ngraph/builder/quantization.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "ngraph/pass/manager.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

TEST(builder, scaled_QMP_unsigned)
{
    vector<uint8_t> a_data = {0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0, 1};
    Shape shape_a{1, 1, 3, 5};
    Shape window_shape{2, 3};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{0, 0};
    Shape padding_above{0, 0};
    Shape shape_r{1, 1, 2, 3};
    auto A = make_shared<op::Parameter>(element::u8, shape_a);
    auto B = op::Constant::create(element::f32, Shape{1}, {0.0f});
    auto C = op::Constant::create(element::f32, Shape{1}, {255.0f});
    auto QMP = ngraph::builder::ScaledQuantizedMaxPool(
        A, window_shape, window_movement_strides, padding_below, padding_above, B, C);
    auto f = make_shared<Function>(NodeVector{QMP}, ParameterVector{A});
    auto backend = runtime::Backend::create("CPU");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::u8, shape_a);
    copy_data(a, a_data);
    auto result = backend->create_tensor(element::u8, shape_r);
    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<uint8_t>{3, 3, 2, 3, 3, 2}), read_vector<uint8_t>(result));
}

TEST(builder, scaled_QMP_signed)
{
    vector<int8_t> a_data = {0, 1, 0, -2, 1, 0, -3, 2, 0, 0, 2, 0, 0, 0, 1};
    Shape shape_a{1, 1, 3, 5};
    Shape window_shape{2, 3};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{0, 0};
    Shape padding_above{0, 0};
    Shape shape_r{1, 1, 2, 3};
    auto A = make_shared<op::Parameter>(element::i8, shape_a);
    auto B = op::Constant::create(element::f32, Shape{1}, {0.0f});
    auto C = op::Constant::create(element::f32, Shape{1}, {127.0f});
    auto QMP = ngraph::builder::ScaledQuantizedMaxPool(
        A, window_shape, window_movement_strides, padding_below, padding_above, B, C);
    auto f = make_shared<Function>(NodeVector{QMP}, ParameterVector{A});
    auto backend = runtime::Backend::create("CPU");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i8, shape_a);
    copy_data(a, a_data);
    auto result = backend->create_tensor(element::i8, shape_r);
    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<int8_t>{2, 2, 2, 2, 2, 2}), read_vector<int8_t>(result));
}

TEST(builder, scaled_QAP_unsigned)
{
    vector<uint8_t> a_data = {0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0, 1};
    Shape shape_a{1, 1, 3, 5};
    Shape window_shape{2, 3};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{0, 0};
    Shape padding_above{0, 0};
    Shape shape_r{1, 1, 2, 3};
    auto A = make_shared<op::Parameter>(element::u8, shape_a);
    auto B = op::Constant::create(element::f32, Shape{1}, {0.0f});
    auto C = op::Constant::create(element::f32, Shape{1}, {255.0f});
    auto QAP = ngraph::builder::ScaledQuantizedAvgPool(
        A, window_shape, window_movement_strides, padding_below, padding_above, false, B, C);
    auto f = make_shared<Function>(NodeVector{QAP}, ParameterVector{A});
    auto backend = runtime::Backend::create("CPU");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::u8, shape_a);
    copy_data(a, a_data);
    auto result = backend->create_tensor(element::u8, shape_r);
    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<uint8_t>{1, 1, 1, 1, 1, 0}), read_vector<uint8_t>(result));
}

TEST(builder, scaled_QAP_signed)
{
    vector<int8_t> a_data = {10, 1, 0, -2, 1, 0, -3, 4, 0, 0, 2, 0, 0, 0, 1};
    Shape shape_a{1, 1, 3, 5};
    Shape window_shape{2, 3};
    auto window_movement_strides = Strides{1, 1};
    Shape padding_below{0, 0};
    Shape padding_above{0, 0};
    Shape shape_r{1, 1, 2, 3};
    auto A = make_shared<op::Parameter>(element::i8, shape_a);
    auto B = op::Constant::create(element::f32, Shape{1}, {0.0f});
    auto C = op::Constant::create(element::f32, Shape{1}, {127.0f});
    auto QAP = ngraph::builder::ScaledQuantizedAvgPool(
        A, window_shape, window_movement_strides, padding_below, padding_above, false, B, C);
    auto f = make_shared<Function>(NodeVector{QAP}, ParameterVector{A});
    auto backend = runtime::Backend::create("CPU");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i8, shape_a);
    copy_data(a, a_data);
    auto result = backend->create_tensor(element::i8, shape_r);
    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<int8_t>{2, 0, 0, 0, 0, 1}), read_vector<int8_t>(result));
}

static void constant_fold(std::shared_ptr<Function> f)
{
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::ConstantFolding>();
    pass_manager.run_passes(f);
}

TEST(builder, scaled_QC)
{
    Shape shape_a{1, 1, 3, 4}; // input shape
    Shape shape_b{1, 1, 3, 3}; // filter shape
    Shape shape_r{1, 1, 3, 4}; // output shape
    vector<uint8_t> a_data = {1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4};
    vector<int8_t> b_data = {1, 2, 3, 4, 5, 0, 0, 1, 2};
    auto A = make_shared<op::Parameter>(element::u8, shape_a);
    auto B = make_shared<op::Parameter>(element::i8, shape_b);
    auto C = op::Constant::create(element::f32, Shape{1}, {0.0f});
    auto D = op::Constant::create(element::f32, Shape{1}, {255.0f});
    auto E = op::Constant::create(element::f32, Shape{1}, {-127.0f});
    auto F = op::Constant::create(element::f32, Shape{1}, {127.0f});
    auto G = op::Constant::create(element::f32, Shape{1}, {22.0f});
    auto H = op::Constant::create(element::f32, Shape{1}, {90.0f});
    auto CV = ngraph::builder::ScaledQuantizedConvolution(A,
                                                          B,
                                                          Strides{1, 1},        // move_strides
                                                          Strides{1, 1},        // filter_dilation
                                                          CoordinateDiff{1, 1}, // below_pads
                                                          CoordinateDiff{1, 1}, // above_pads
                                                          Strides{1, 1},        // data_dilation
                                                          C,
                                                          D,
                                                          E,
                                                          F,
                                                          G,
                                                          H);
    auto f = make_shared<Function>(NodeVector{CV}, ParameterVector{A, B});
    constant_fold(f);

    auto backend = runtime::Backend::create("CPU");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::u8, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::i8, shape_b);
    copy_data(b, b_data);
    auto result = backend->create_tensor(element::i8, shape_r);
    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});
    EXPECT_EQ((vector<int8_t>{31, 48, 42, 45, 54, 102, 127, 61, 47, 74, 61, 55}),
              read_vector<int8_t>(result));
}

TEST(builder, scaled_QC_with_relu)
{
    Shape shape_a{1, 1, 3, 3}; // input shape
    Shape shape_b{1, 1, 3, 3}; // filter shape
    Shape shape_r{1, 1, 3, 3}; // output shape
    vector<uint8_t> a_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    vector<int8_t> b_data = {1, 2, 1, 0, 0, 0, -1, -2, -1};
    auto A = make_shared<op::Parameter>(element::u8, shape_a);
    auto B = make_shared<op::Parameter>(element::i8, shape_b);
    auto C = op::Constant::create(element::f32, Shape{1}, {0.0f});
    auto D = op::Constant::create(element::f32, Shape{1}, {255.0f});
    auto E = op::Constant::create(element::f32, Shape{1}, {-127.0f});
    auto F = op::Constant::create(element::f32, Shape{1}, {127.0f});
    auto G = op::Constant::create(element::f32, Shape{1}, {20.0f});
    auto H = op::Constant::create(element::f32, Shape{1}, {-24.0f});
    auto CV = ngraph::builder::ScaledQuantizedConvolutionRelu(A,
                                                              B,
                                                              Strides{1, 1}, // move_strides
                                                              Strides{1, 1}, // filter_dilation
                                                              CoordinateDiff{1, 1}, // below_pads
                                                              CoordinateDiff{1, 1}, // above_pads
                                                              Strides{1, 1},        // data_dilation
                                                              C,
                                                              D,
                                                              E,
                                                              F,
                                                              G,
                                                              H);
    auto f = make_shared<Function>(NodeVector{CV}, ParameterVector{A, B});
    constant_fold(f);
    auto backend = runtime::Backend::create("CPU");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::u8, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::i8, shape_b);
    copy_data(b, b_data);
    auto result = backend->create_tensor(element::u8, shape_r);
    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});
    EXPECT_EQ((vector<uint8_t>{0, 0, 0, 0, 0, 0, 138, 212, 181}), read_vector<uint8_t>(result));
}

TEST(builder, scaled_QC_with_bias)
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
    auto C = op::Constant::create(element::f32, Shape{1}, {0.0f});
    auto D = op::Constant::create(element::f32, Shape{1}, {255.0f});
    auto E = op::Constant::create(element::f32, Shape{1}, {-127.0f});
    auto F = op::Constant::create(element::f32, Shape{1}, {127.0f});
    auto G = op::Constant::create(element::f32, Shape{1}, {22.0f});
    auto H = op::Constant::create(element::f32, Shape{1}, {90.0f});
    auto CV = ngraph::builder::ScaledQuantizedConvolutionBias(A,
                                                              B,
                                                              Bias,
                                                              Strides{1, 1}, // move_strides
                                                              Strides{1, 1}, // filter_dilation
                                                              CoordinateDiff{1, 1}, // below_pads
                                                              CoordinateDiff{1, 1}, // above_pads
                                                              Strides{1, 1},        // data_dilation
                                                              C,
                                                              D,
                                                              E,
                                                              F,
                                                              G,
                                                              H);
    auto f = make_shared<Function>(NodeVector{CV}, ParameterVector{A, B, Bias});
    constant_fold(f);
    auto backend = runtime::Backend::create("CPU");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::u8, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::i8, shape_b);
    copy_data(b, b_data);
    auto c = backend->create_tensor(element::i32, Shape{1});
    copy_data(c, c_data);
    auto result = backend->create_tensor(element::i8, shape_r);
    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b, c});
    EXPECT_EQ((vector<int8_t>{38, 55, 50, 52, 61, 109, 127, 68, 54, 81, 68, 62}),
              read_vector<int8_t>(result));
}

TEST(builder, scaled_QC_with_bias_and_relu)
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
    auto C = op::Constant::create(element::f32, Shape{1}, {0.0f});
    auto D = op::Constant::create(element::f32, Shape{1}, {255.0f});
    auto E = op::Constant::create(element::f32, Shape{1}, {-127.0f});
    auto F = op::Constant::create(element::f32, Shape{1}, {127.0f});
    auto G = op::Constant::create(element::f32, Shape{1}, {20.0f});
    auto H = op::Constant::create(element::f32, Shape{1}, {-24.0f});
    auto CV = ngraph::builder::ScaledQuantizedConvolutionBias(A,
                                                              B,
                                                              Bias,
                                                              Strides{1, 1}, // move_strides
                                                              Strides{1, 1}, // filter_dilation
                                                              CoordinateDiff{1, 1}, // below_pads
                                                              CoordinateDiff{1, 1}, // above_pads
                                                              Strides{1, 1},        // data_dilation
                                                              C,
                                                              D,
                                                              E,
                                                              F,
                                                              G,
                                                              H,
                                                              true);
    auto f = make_shared<Function>(NodeVector{CV}, ParameterVector{A, B, Bias});
    constant_fold(f);
    auto backend = runtime::Backend::create("CPU");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::u8, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::i8, shape_b);
    copy_data(b, b_data);
    auto c = backend->create_tensor(element::i32, Shape{1});
    copy_data(c, c_data);
    auto result = backend->create_tensor(element::u8, shape_r);
    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b, c});
    EXPECT_EQ((vector<uint8_t>{0, 0, 0, 0, 0, 0, 191, 255, 234}), read_vector<uint8_t>(result));
}

TEST(builder, scaled_QC_with_bias_add_and_relu)
{
    Shape shape_a{1, 1, 3, 4}; // input shape
    Shape shape_b{1, 1, 3, 3}; // filter shape
    Shape shape_r{1, 1, 3, 4}; // output shape
    vector<uint8_t> a_data = {1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4};
    vector<int8_t> b_data = {1, 2, 3, 4, 5, 0, 0, 1, 2};
    vector<int32_t> c_data = {5};
    vector<uint8_t> conv_2_data = {1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4};
    auto A = make_shared<op::Parameter>(element::u8, shape_a);
    auto B = make_shared<op::Parameter>(element::i8, shape_b);
    auto Add = make_shared<op::Parameter>(element::u8, shape_a);
    auto Bias = make_shared<op::Parameter>(element::i32, Shape{1});
    auto C = op::Constant::create(element::f32, Shape{}, {0.0f});
    auto D = op::Constant::create(element::f32, Shape{}, {255.0f});
    auto E = op::Constant::create(element::f32, Shape{}, {-127.0f});
    auto F = op::Constant::create(element::f32, Shape{}, {127.0f});
    auto G = op::Constant::create(element::f32, Shape{}, {22.0f});
    auto H = op::Constant::create(element::f32, Shape{}, {90.0f});
    auto I = op::Constant::create(element::f32, Shape{}, {22.0f});
    auto J = op::Constant::create(element::f32, Shape{}, {180.0f});
    auto CV = ngraph::builder::ScaledQuantizedConvolutionBiasAdd(A,
                                                                 B,
                                                                 Bias,
                                                                 Add,
                                                                 Strides{1, 1}, // move_strides
                                                                 Strides{1, 1}, // filter_dilation
                                                                 CoordinateDiff{1, 1}, // below_pads
                                                                 CoordinateDiff{1, 1}, // above_pads
                                                                 Strides{1, 1}, // data_dilation
                                                                 C,
                                                                 D,
                                                                 E,
                                                                 F,
                                                                 G,
                                                                 H,
                                                                 I,
                                                                 J,
                                                                 true);
    auto f = make_shared<Function>(NodeVector{CV}, ParameterVector{A, B, Bias, Add});
    constant_fold(f);
    auto backend = runtime::Backend::create("CPU");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::u8, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::i8, shape_b);
    copy_data(b, b_data);
    auto c = backend->create_tensor(element::i32, Shape{1});
    copy_data(c, c_data);
    auto d = backend->create_tensor(element::u8, shape_a);
    copy_data(d, conv_2_data);
    auto result = backend->create_tensor(element::u8, shape_r);
    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b, c, d});
    EXPECT_EQ((vector<uint8_t>{78, 114, 105, 113, 132, 230, 255, 136, 110, 165, 142, 133}),
              read_vector<uint8_t>(result));
}

TEST(builder, scaled_QC_with_bias_signed_add_and_relu)
{
    Shape shape_a{1, 1, 3, 4}; // input shape
    Shape shape_b{1, 1, 3, 3}; // filter shape
    Shape shape_r{1, 1, 3, 4}; // output shape
    vector<uint8_t> a_data = {1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4};
    vector<int8_t> b_data = {1, 2, 3, 4, 5, 0, 0, 1, 2};
    vector<int32_t> c_data = {5};
    vector<int8_t> conv_2_data = {-1, -2, -3, -4, -5, -6, -10, 0, 1, 2, 3, 4};
    auto A = make_shared<op::Parameter>(element::u8, shape_a);
    auto B = make_shared<op::Parameter>(element::i8, shape_b);
    auto Add = make_shared<op::Parameter>(element::i8, shape_a);
    auto Bias = make_shared<op::Parameter>(element::i32, Shape{1});
    auto C = op::Constant::create(element::f32, Shape{}, {0.0f});
    auto D = op::Constant::create(element::f32, Shape{}, {255.0f});
    auto E = op::Constant::create(element::f32, Shape{}, {-127.0f});
    auto F = op::Constant::create(element::f32, Shape{}, {127.0f});
    auto G = op::Constant::create(element::f32, Shape{}, {22.0f});
    auto H = op::Constant::create(element::f32, Shape{}, {90.0f});
    auto I = op::Constant::create(element::f32, Shape{}, {22.0f});
    auto J = op::Constant::create(element::f32, Shape{}, {90.0f});
    auto CV =
        ngraph::builder::ScaledQuantizedConvolutionBiasSignedAdd(A,
                                                                 B,
                                                                 Bias,
                                                                 Add,
                                                                 Strides{1, 1}, // move_strides
                                                                 Strides{1, 1}, // filter_dilation
                                                                 CoordinateDiff{1, 1}, // below_pads
                                                                 CoordinateDiff{1, 1}, // above_pads
                                                                 Strides{1, 1}, // data_dilation
                                                                 C,
                                                                 D,
                                                                 E,
                                                                 F,
                                                                 G,
                                                                 H,
                                                                 I,
                                                                 J,
                                                                 true);
    auto f = make_shared<Function>(NodeVector{CV}, ParameterVector{A, B, Bias, Add});
    constant_fold(f);
    auto backend = runtime::Backend::create("CPU");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::u8, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::i8, shape_b);
    copy_data(b, b_data);
    auto c = backend->create_tensor(element::i32, Shape{1});
    copy_data(c, c_data);
    auto d = backend->create_tensor(element::i8, shape_a);
    copy_data(d, conv_2_data);
    auto result = backend->create_tensor(element::u8, shape_r);
    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b, c, d});
    EXPECT_EQ((vector<uint8_t>{76, 110, 99, 105, 122, 218, 255, 136, 110, 165, 142, 133}),
              read_vector<uint8_t>(result));
}

TEST(builder, scaled_QC_with_f32_bias_and_relu)
{
    Shape shape_a{1, 1, 3, 3}; // input shape
    Shape shape_b{1, 1, 3, 3}; // filter shape
    Shape shape_r{1, 1, 3, 3}; // output shape
    vector<uint8_t> a_data = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    vector<int8_t> b_data = {1, 2, 1, 0, 0, 0, -1, -2, -1};
    vector<float> c_data = {5};
    auto A = make_shared<op::Parameter>(element::u8, shape_a);
    auto B = make_shared<op::Parameter>(element::i8, shape_b);
    auto Bias = make_shared<op::Parameter>(element::f32, Shape{1});
    auto C = op::Constant::create(element::f32, Shape{}, {0.0f});
    auto D = op::Constant::create(element::f32, Shape{}, {255.0f});
    auto E = op::Constant::create(element::f32, Shape{}, {-127.0f});
    auto F = op::Constant::create(element::f32, Shape{}, {127.0f});
    auto G = op::Constant::create(element::f32, Shape{}, {20.0f});
    auto H = op::Constant::create(element::f32, Shape{}, {-24.0f});
    auto CV = ngraph::builder::ScaledQuantizedConvolutionBias(A,
                                                              B,
                                                              Bias,
                                                              Strides{1, 1}, // move_strides
                                                              Strides{1, 1}, // filter_dilation
                                                              CoordinateDiff{1, 1}, // below_pads
                                                              CoordinateDiff{1, 1}, // above_pads
                                                              Strides{1, 1},        // data_dilation
                                                              C,
                                                              D,
                                                              E,
                                                              F,
                                                              G,
                                                              H,
                                                              true);
    auto f = make_shared<Function>(NodeVector{CV}, ParameterVector{A, B, Bias});
    constant_fold(f);
    auto backend = runtime::Backend::create("CPU");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::u8, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::i8, shape_b);
    copy_data(b, b_data);
    auto c = backend->create_tensor(element::f32, Shape{1});
    copy_data(c, c_data);
    auto result = backend->create_tensor(element::u8, shape_r);
    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b, c});
    EXPECT_EQ((vector<uint8_t>{0, 0, 0, 0, 0, 0, 191, 255, 234}), read_vector<uint8_t>(result));
}

TEST(builder, scaled_Q_unsigned)
{
    vector<float> a_data = {-255.0, 0.0, 1.0, 1.25, 1.75, 64.0, 127.0, 500.0};
    Shape shape_a{8};
    AxisSet quantization_axes;
    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = op::Constant::create(element::f32, Shape{}, {-255.0f});
    auto C = op::Constant::create(element::f32, Shape{}, {127.0f});
    auto QT = ngraph::builder::ScaledQuantize(A, B, C, element::u8, quantization_axes, round_mode);
    auto f = make_shared<Function>(NodeVector{QT}, ParameterVector{A});
    constant_fold(f);
    auto backend = runtime::Backend::create("CPU");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, a_data);
    auto result = backend->create_tensor(element::u8, shape_a);
    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<uint8_t>{0, 0, 1, 1, 2, 64, 127, 255}), read_vector<uint8_t>(result));
}

TEST(builder, dynamic_scaled_Q_unsigned)
{
    vector<float> a_data = {-255.0, 0.0, 1.0, 1.25, 1.75, 64.0, 127.0, 500.0};
    Shape shape_a{8};
    AxisSet quantization_axes;
    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, Shape{});
    auto C = make_shared<op::Parameter>(element::f32, Shape{});
    auto QT = ngraph::builder::ScaledQuantize(A, B, C, element::u8, quantization_axes, round_mode);
    auto f = make_shared<Function>(NodeVector{QT}, ParameterVector{A, B, C});
    auto backend = runtime::Backend::create("CPU");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    auto b = backend->create_tensor(element::f32, Shape{});
    auto c = backend->create_tensor(element::f32, Shape{});
    copy_data(a, a_data);
    copy_data(b, vector<float>{-255.0f});
    copy_data(c, vector<float>{127.0f});
    auto result = backend->create_tensor(element::u8, shape_a);
    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b, c});
    EXPECT_EQ((vector<uint8_t>{0, 0, 1, 1, 2, 64, 127, 255}), read_vector<uint8_t>(result));
}

TEST(builder, scaled_Q_signed)
{
    vector<float> a_data = {-127.0, 0.0, 1.0, 3.0, 5.0, 64.0, 127.0, 500.0};
    Shape shape_a{8};
    AxisSet quantization_axes;
    op::Quantize::RoundMode round_mode = op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN;
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = op::Constant::create(element::f32, Shape{}, {-127.0f});
    auto C = op::Constant::create(element::f32, Shape{}, {127.0f});
    auto QT = ngraph::builder::ScaledQuantize(A, B, C, element::i8, quantization_axes, round_mode);
    auto f = make_shared<Function>(NodeVector{QT}, ParameterVector{A});
    constant_fold(f);
    auto backend = runtime::Backend::create("CPU");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, a_data);
    auto result = backend->create_tensor(element::i8, shape_a);
    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<int8_t>{-127, 0, 1, 3, 5, 64, 127, 127}), read_vector<int8_t>(result));
}

TEST(builder, scaled_DQ_signed)
{
    vector<int8_t> a_data = {42};
    AxisSet quantization_axes;
    auto A = make_shared<op::Parameter>(element::i8, Shape{1});
    auto B = op::Constant::create(element::f32, Shape{}, {-1.0f});
    auto C = op::Constant::create(element::f32, Shape{}, {300.0f});
    auto r = ngraph::builder::ScaledDequantize(A, B, C, element::f32, quantization_axes);
    auto f = make_shared<Function>(r, ParameterVector{A});
    constant_fold(f);
    auto backend = runtime::Backend::create("CPU");
    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i8, Shape{1});
    copy_data(a, a_data);
    auto result = backend->create_tensor(element::f32, Shape{1});
    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{99.212601}), read_vector<float>(result));
}
