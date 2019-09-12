//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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
#include <iterator>
#include <limits>
#include <random>
#include <string>

#include "gtest/gtest.h"
#include "ngraph/check.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, elu)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{3, 2});
    auto elu = make_shared<op::Elu>(A, 0.5f);
    auto function = make_shared<Function>(NodeVector{elu}, ParameterVector{A});

    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input(vector<float>{-2.f, 3.f, -2.f, 1.f, -1.f, 0.f});
    test_case.add_expected_output(
        vector<float>{-0.432332358f, 3.f, -0.432332358f, 1.f, -0.316060279f, 0.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, elu_negative_alpha)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{3, 2});
    auto elu = make_shared<op::Elu>(A, -1.f);
    auto function = make_shared<Function>(NodeVector{elu}, ParameterVector{A});

    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input(vector<float>{-2.f, 3.f, -2.f, 1.f, -1.f, 0.f});
    test_case.add_expected_output(
        vector<float>{0.864664717f, 3.f, 0.864664717f, 1.f, 0.632120559f, 0.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu)
{
    Shape shape{3, 2};
    Shape rshape{3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, rshape);
    auto prelu = make_shared<op::PRelu>(A, B);
    auto f0 = make_shared<Function>(NodeVector{prelu}, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{-2, 3, -2, 1, -1, 0});
    auto b = backend->create_tensor(element::f32, rshape);
    copy_data(b, vector<float>{0, 0.5, 1});
    auto result0 = backend->create_tensor(element::f32, shape);
    auto handle = backend->compile(f0);
    handle->call_with_validate({result0}, {a, b});
    vector<float> expected{0, 3, -1, 1, -1, 0};
    EXPECT_EQ(expected, read_vector<float>(result0));
}

NGRAPH_TEST(${BACKEND_NAME}, hardsigmoid)
{
    Shape shape{2, 7};
    float alpha = 0.125f;
    float beta = 0.642f;

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto hardsigmoid = make_shared<op::HardSigmoid>(A, alpha, beta);
    auto f0 = make_shared<Function>(NodeVector{hardsigmoid}, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Prepare input and expected output data
    vector<float> input_data{-1.f,
                             0.f,
                             1.f,
                             -100.f,
                             100.f,
                             -3.1234567f,
                             5.876543f,
                             7.13245364f,
                             numeric_limits<float>::max(),
                             numeric_limits<float>::lowest(),
                             numeric_limits<float>::min(),
                             numeric_limits<float>::infinity(),
                             numeric_limits<float>::min() / 16.f,
                             -numeric_limits<float>::min() / 16.f};

    auto impl = [alpha, beta](float val) { return min(max(alpha * val + beta, 0.f), 1.f); };
    vector<float> expected_output;
    transform(begin(input_data), end(input_data), back_inserter(expected_output), impl);

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, input_data);
    auto result0 = backend->create_tensor(element::f32, shape);
    auto handle = backend->compile(f0);
    handle->call_with_validate({result0}, {a});

    EXPECT_TRUE(test::all_close_f(expected_output, read_vector<float>(result0)));
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_shared_slope)
{
    Shape shape{3, 2};
    Shape rshape{};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, rshape);
    auto prelu = make_shared<op::PRelu>(A, B);
    auto f0 = make_shared<Function>(NodeVector{prelu}, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{-2, 3, -2, 1, -1, 0});
    auto b = backend->create_tensor(element::f32, rshape);
    copy_data(b, vector<float>{0.5});
    auto result0 = backend->create_tensor(element::f32, shape);
    auto handle = backend->compile(f0);
    handle->call_with_validate({result0}, {a, b});
    vector<float> expected{-1, 3, -1, 1, -0.5, 0};
    EXPECT_EQ(expected, read_vector<float>(result0));
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_negative_slope)
{
    Shape shape{3, 2};
    Shape rshape{};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, rshape);
    auto prelu = make_shared<op::PRelu>(A, B);
    auto f0 = make_shared<Function>(NodeVector{prelu}, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{-2, 3, -2, 1, -1, 0});
    auto b = backend->create_tensor(element::f32, rshape);
    copy_data(b, vector<float>{-0.5});
    auto result0 = backend->create_tensor(element::f32, shape);
    auto handle = backend->compile(f0);
    handle->call_with_validate({result0}, {a, b});
    vector<float> expected{1, 3, 1, 1, 0.5, 0};
    EXPECT_EQ(expected, read_vector<float>(result0));
}

NGRAPH_TEST(${BACKEND_NAME}, conv_bias_1d)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 3, 2});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{2, 3, 1});
    auto bias = make_shared<op::Parameter>(element::f32, Shape{2});
    auto conv_bias = make_shared<op::ConvolutionBias>(data, filters, bias);
    auto f0 = make_shared<Function>(NodeVector{conv_bias}, ParameterVector{data, filters, bias});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, Shape{1, 3, 2});
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto b = backend->create_tensor(element::f32, Shape{2, 3, 1});
    copy_data(b, vector<float>{1, 2, 3, 4, 5, 6});
    auto c = backend->create_tensor(element::f32, Shape{2});
    copy_data(c, vector<float>{1, 2});
    auto result0 = backend->create_tensor(element::f32, conv_bias->get_shape());
    auto handle = backend->compile(f0);
    handle->call_with_validate({result0}, {a, b, c});
    vector<float> expected{23, 29, 51, 66};
    EXPECT_EQ(expected, read_vector<float>(result0));
}

NGRAPH_TEST(${BACKEND_NAME}, conv_bias_2d)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 3, 2, 2});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{2, 3, 1, 1});
    auto bias = make_shared<op::Parameter>(element::f32, Shape{2});
    auto conv_bias = make_shared<op::ConvolutionBias>(data, filters, bias);
    auto f0 = make_shared<Function>(NodeVector{conv_bias}, ParameterVector{data, filters, bias});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, Shape{1, 3, 2, 2});
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto b = backend->create_tensor(element::f32, Shape{2, 3, 1, 1});
    copy_data(b, vector<float>{1, 2, 3, 4, 5, 6});
    auto c = backend->create_tensor(element::f32, Shape{2});
    copy_data(c, vector<float>{1, 2});
    auto result0 = backend->create_tensor(element::f32, conv_bias->get_shape());
    auto handle = backend->compile(f0);
    handle->call_with_validate({result0}, {a, b, c});
    vector<float> expected{39, 45, 51, 57, 85, 100, 115, 130};
    EXPECT_EQ(expected, read_vector<float>(result0));
}

NGRAPH_TEST(${BACKEND_NAME}, conv_bias_3d)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 3, 1, 2, 2});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{2, 3, 1, 1, 1});
    auto bias = make_shared<op::Parameter>(element::f32, Shape{2});
    auto conv_bias = make_shared<op::ConvolutionBias>(data, filters, bias);
    auto f0 = make_shared<Function>(NodeVector{conv_bias}, ParameterVector{data, filters, bias});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, Shape{1, 3, 1, 2, 2});
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto b = backend->create_tensor(element::f32, Shape{2, 3, 1, 1, 1});
    copy_data(b, vector<float>{1, 2, 3, 4, 5, 6});
    auto c = backend->create_tensor(element::f32, Shape{2});
    copy_data(c, vector<float>{1, 2});
    auto result0 = backend->create_tensor(element::f32, conv_bias->get_shape());
    auto handle = backend->compile(f0);
    handle->call_with_validate({result0}, {a, b, c});
    vector<float> expected{39, 45, 51, 57, 85, 100, 115, 130};
    EXPECT_EQ(expected, read_vector<float>(result0));
}

NGRAPH_TEST(${BACKEND_NAME}, conv_bias_bprop_2d)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 3, 2, 2});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{2, 3, 1, 1});
    auto bias = make_shared<op::Parameter>(element::f32, Shape{2});
    auto delta = make_shared<op::Parameter>(element::f32, Shape{1, 2, 2, 2});
    auto conv_bprop = make_shared<op::ConvolutionBiasBackpropFiltersBias>(data,
                                                                          filters->get_shape(),
                                                                          bias->get_shape(),
                                                                          delta,
                                                                          Strides{1, 1},
                                                                          Strides{1, 1},
                                                                          CoordinateDiff{0, 0},
                                                                          CoordinateDiff{0, 0},
                                                                          Strides{1, 1});
    auto goe0 = make_shared<op::GetOutputElement>(conv_bprop, 0);
    auto goe1 = make_shared<op::GetOutputElement>(conv_bprop, 1);
    auto f0 = make_shared<Function>(NodeVector{goe0, goe1}, ParameterVector{data, delta});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, Shape{1, 3, 2, 2});
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto b = backend->create_tensor(element::f32, Shape{1, 2, 2, 2});
    copy_data(b, vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    auto result0 = backend->create_tensor(element::f32, filters->get_shape());
    auto result1 = backend->create_tensor(element::f32, bias->get_shape());
    auto handle = backend->compile(f0);
    handle->call_with_validate({result0, result1}, {a, b});
    vector<float> expected0{30, 70, 110, 70, 174, 278};
    vector<float> expected1{10, 26};
    EXPECT_EQ(expected0, read_vector<float>(result0));
    EXPECT_EQ(expected1, read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, conv_bias_add_2d)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 3, 2, 2});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{2, 3, 1, 1});
    auto bias = make_shared<op::Parameter>(element::f32, Shape{2});
    auto add = make_shared<op::Parameter>(element::f32, Shape{1, 2, 2, 2});
    auto conv_bias = make_shared<op::ConvolutionBias>(data, filters, bias);
    auto conv_bias_add = make_shared<op::ConvolutionBiasAdd>(conv_bias, add);
    auto f0 =
        make_shared<Function>(NodeVector{conv_bias_add}, ParameterVector{data, filters, bias, add});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, Shape{1, 3, 2, 2});
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto b = backend->create_tensor(element::f32, Shape{2, 3, 1, 1});
    copy_data(b, vector<float>{1, 2, 3, 4, 5, 6});
    auto c = backend->create_tensor(element::f32, Shape{2});
    copy_data(c, vector<float>{1, 2});
    auto d = backend->create_tensor(element::f32, Shape{1, 2, 2, 2});
    copy_data(d, vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    auto result0 = backend->create_tensor(element::f32, conv_bias_add->get_shape());
    auto handle = backend->compile(f0);
    handle->call_with_validate({result0}, {a, b, c, d});
    vector<float> expected{40, 47, 54, 61, 90, 106, 122, 138};
    EXPECT_EQ(expected, read_vector<float>(result0));
}

NGRAPH_TEST(${BACKEND_NAME}, group_conv)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 4, 2, 2});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{2, 2, 1, 1});
    auto group_conv = make_shared<op::GroupConvolution>(data,
                                                        filters,
                                                        Strides{1, 1},
                                                        Strides{1, 1},
                                                        CoordinateDiff{0, 0},
                                                        CoordinateDiff{0, 0},
                                                        Strides{1, 1},
                                                        2);
    auto f0 = make_shared<Function>(NodeVector{group_conv}, ParameterVector{data, filters});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, Shape{1, 4, 2, 2});
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto b = backend->create_tensor(element::f32, Shape{2, 2, 1, 1});
    copy_data(b, vector<float>{1, 2, 3, 4});
    auto result0 = backend->create_tensor(element::f32, Shape{1, 2, 2, 2});
    auto handle = backend->compile(f0);
    handle->call_with_validate({result0}, {a, b});
    vector<float> expected{11, 14, 17, 20, 79, 86, 93, 100};
    EXPECT_EQ(expected, read_vector<float>(result0));
}

NGRAPH_TEST(${BACKEND_NAME}, group_conv_striding)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 4, 2, 2});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{2, 2, 1, 1});
    auto group_conv = make_shared<op::GroupConvolution>(data,
                                                        filters,
                                                        Strides{2, 2},
                                                        Strides{1, 1},
                                                        CoordinateDiff{0, 0},
                                                        CoordinateDiff{0, 0},
                                                        Strides{1, 1},
                                                        2);
    auto f0 = make_shared<Function>(NodeVector{group_conv}, ParameterVector{data, filters});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, Shape{1, 4, 2, 2});
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto b = backend->create_tensor(element::f32, Shape{2, 2, 1, 1});
    copy_data(b, vector<float>{1, 2, 3, 4});
    auto result0 = backend->create_tensor(element::f32, Shape{1, 2, 1, 1});
    auto handle = backend->compile(f0);
    handle->call_with_validate({result0}, {a, b});
    vector<float> expected{11, 79};
    EXPECT_EQ(expected, read_vector<float>(result0));
}

NGRAPH_TEST(${BACKEND_NAME}, group_conv_window_dilation)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 4, 2, 2});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{2, 2, 1, 1});
    auto group_conv = make_shared<op::GroupConvolution>(data,
                                                        filters,
                                                        Strides{1, 1},
                                                        Strides{2, 2},
                                                        CoordinateDiff{0, 0},
                                                        CoordinateDiff{0, 0},
                                                        Strides{1, 1},
                                                        2);
    auto f0 = make_shared<Function>(NodeVector{group_conv}, ParameterVector{data, filters});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, Shape{1, 4, 2, 2});
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto b = backend->create_tensor(element::f32, Shape{2, 2, 1, 1});
    copy_data(b, vector<float>{1, 2, 3, 4});
    auto result0 = backend->create_tensor(element::f32, Shape{1, 2, 2, 2});
    auto handle = backend->compile(f0);
    handle->call_with_validate({result0}, {a, b});
    vector<float> expected{11, 14, 17, 20, 79, 86, 93, 100};
    EXPECT_EQ(expected, read_vector<float>(result0));
}

NGRAPH_TEST(${BACKEND_NAME}, group_conv_data_dilation)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 4, 2, 2});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{2, 2, 1, 1});
    auto group_conv = make_shared<op::GroupConvolution>(data,
                                                        filters,
                                                        Strides{1, 1},
                                                        Strides{1, 1},
                                                        CoordinateDiff{0, 0},
                                                        CoordinateDiff{0, 0},
                                                        Strides{2, 2},
                                                        2);
    auto f0 = make_shared<Function>(NodeVector{group_conv}, ParameterVector{data, filters});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, Shape{1, 4, 2, 2});
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto b = backend->create_tensor(element::f32, Shape{2, 2, 1, 1});
    copy_data(b, vector<float>{1, 2, 3, 4});
    auto result0 = backend->create_tensor(element::f32, Shape{1, 2, 3, 3});
    auto handle = backend->compile(f0);
    handle->call_with_validate({result0}, {a, b});
    vector<float> expected{11, 0, 14, 0, 0, 0, 17, 0, 20, 79, 0, 86, 0, 0, 0, 93, 0, 100};
    EXPECT_EQ(expected, read_vector<float>(result0));
}

NGRAPH_TEST(${BACKEND_NAME}, group_conv_padding)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 4, 2, 2});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{2, 2, 1, 1});
    auto group_conv = make_shared<op::GroupConvolution>(data,
                                                        filters,
                                                        Strides{1, 1},
                                                        Strides{1, 1},
                                                        CoordinateDiff{1, 0},
                                                        CoordinateDiff{0, 1},
                                                        Strides{1, 1},
                                                        2);
    auto f0 = make_shared<Function>(NodeVector{group_conv}, ParameterVector{data, filters});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, Shape{1, 4, 2, 2});
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto b = backend->create_tensor(element::f32, Shape{2, 2, 1, 1});
    copy_data(b, vector<float>{1, 2, 3, 4});
    auto result0 = backend->create_tensor(element::f32, Shape{1, 2, 3, 3});
    auto handle = backend->compile(f0);
    handle->call_with_validate({result0}, {a, b});
    vector<float> expected{0, 0, 0, 11, 14, 0, 17, 20, 0, 0, 0, 0, 79, 86, 0, 93, 100, 0};
    EXPECT_EQ(expected, read_vector<float>(result0));
}

NGRAPH_TEST(${BACKEND_NAME}, group_conv_padding_and_window_dilation)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 4, 2, 2});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{2, 2, 1, 1});
    auto group_conv = make_shared<op::GroupConvolution>(data,
                                                        filters,
                                                        Strides{1, 1},
                                                        Strides{2, 2},
                                                        CoordinateDiff{1, 0},
                                                        CoordinateDiff{0, 1},
                                                        Strides{1, 1},
                                                        2);
    auto f0 = make_shared<Function>(NodeVector{group_conv}, ParameterVector{data, filters});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, Shape{1, 4, 2, 2});
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto b = backend->create_tensor(element::f32, Shape{2, 2, 1, 1});
    copy_data(b, vector<float>{1, 2, 3, 4});
    auto result0 = backend->create_tensor(element::f32, Shape{1, 2, 3, 3});
    auto handle = backend->compile(f0);
    handle->call_with_validate({result0}, {a, b});
    vector<float> expected{0, 0, 0, 11, 14, 0, 17, 20, 0, 0, 0, 0, 79, 86, 0, 93, 100, 0};
    EXPECT_EQ(expected, read_vector<float>(result0));
}

NGRAPH_TEST(${BACKEND_NAME}, group_conv_input_shape_variation)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 4, 4, 1});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{2, 2, 1, 1});
    auto group_conv = make_shared<op::GroupConvolution>(data,
                                                        filters,
                                                        Strides{1, 1},
                                                        Strides{2, 2},
                                                        CoordinateDiff{1, 0},
                                                        CoordinateDiff{0, 1},
                                                        Strides{1, 1},
                                                        2);
    auto f0 = make_shared<Function>(NodeVector{group_conv}, ParameterVector{data, filters});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, Shape{1, 4, 4, 1});
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto b = backend->create_tensor(element::f32, Shape{2, 2, 1, 1});
    copy_data(b, vector<float>{1, 2, 3, 4});
    auto result0 = backend->create_tensor(element::f32, Shape{1, 2, 5, 2});
    auto handle = backend->compile(f0);
    handle->call_with_validate({result0}, {a, b});
    vector<float> expected{0, 0, 11, 0, 14, 0, 17, 0, 20, 0, 0, 0, 79, 0, 86, 0, 93, 0, 100, 0};
    EXPECT_EQ(expected, read_vector<float>(result0));
}

NGRAPH_TEST(${BACKEND_NAME}, group_conv_input_data_variation)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 4, 3, 3});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{2, 2, 1, 1});
    auto group_conv = make_shared<op::GroupConvolution>(data,
                                                        filters,
                                                        Strides{1, 1},
                                                        Strides{2, 2},
                                                        CoordinateDiff{1, 0},
                                                        CoordinateDiff{0, 1},
                                                        Strides{1, 1},
                                                        2);
    auto f0 = make_shared<Function>(NodeVector{group_conv}, ParameterVector{data, filters});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, Shape{1, 4, 3, 3});
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                               25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36});
    auto b = backend->create_tensor(element::f32, Shape{2, 2, 1, 1});
    copy_data(b, vector<float>{1, 2, 3, 4});
    auto result0 = backend->create_tensor(element::f32, Shape{1, 2, 4, 4});
    auto handle = backend->compile(f0);
    handle->call_with_validate({result0}, {a, b});
    vector<float> expected{0, 0, 0, 0, 21,  24,  27,  0, 30,  33,  36,  0, 39,  42,  45,  0,
                           0, 0, 0, 0, 169, 176, 183, 0, 190, 197, 204, 0, 211, 218, 225, 0};
    EXPECT_EQ(expected, read_vector<float>(result0));
}

NGRAPH_TEST(${BACKEND_NAME}, group_conv_groups_included_in_shape)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 4, 2, 2});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{2, 1, 2, 1, 1});
    auto group_conv = make_shared<op::GroupConvolution>(data,
                                                        filters,
                                                        Strides{1, 1},
                                                        Strides{1, 1},
                                                        CoordinateDiff{0, 0},
                                                        CoordinateDiff{0, 0},
                                                        Strides{1, 1});
    auto f0 = make_shared<Function>(NodeVector{group_conv}, ParameterVector{data, filters});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, Shape{1, 4, 2, 2});
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto b = backend->create_tensor(element::f32, Shape{2, 1, 2, 1, 1});
    copy_data(b, vector<float>{1, 2, 3, 4});
    auto result0 = backend->create_tensor(element::f32, Shape{1, 2, 2, 2});
    auto handle = backend->compile(f0);
    handle->call_with_validate({result0}, {a, b});
    vector<float> expected{11, 14, 17, 20, 79, 86, 93, 100};
    EXPECT_EQ(expected, read_vector<float>(result0));
}

NGRAPH_TEST(${BACKEND_NAME}, space_to_depth)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 2, 4, 4});
    auto space_to_depth = make_shared<op::SpaceToDepth>(A, 2);
    auto function = make_shared<Function>(NodeVector{space_to_depth}, ParameterVector{A});

    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({0.f,  1.f,  2.f,  3.f,  4.f,  5.f,  6.f,  7.f,  8.f,  9.f,  10.f,
                                11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f,
                                22.f, 23.f, 24.f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 31.f});
    test_case.add_expected_output<float>(Shape{1, 8, 2, 2},
                                         {
                                             0.f, 2.f, 8.f,  10.f, 16.f, 18.f, 24.f, 26.f,
                                             1.f, 3.f, 9.f,  11.f, 17.f, 19.f, 25.f, 27.f,
                                             4.f, 6.f, 12.f, 14.f, 20.f, 22.f, 28.f, 30.f,
                                             5.f, 7.f, 13.f, 15.f, 21.f, 23.f, 29.f, 31.f,
                                         });
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, depth_to_space)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 8, 2, 2});
    auto depth_to_space = make_shared<op::DepthToSpace>(A, 2);
    auto function = make_shared<Function>(NodeVector{depth_to_space}, ParameterVector{A});

    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({
        0.f, 2.f, 8.f,  10.f, 16.f, 18.f, 24.f, 26.f, 1.f, 3.f, 9.f,  11.f, 17.f, 19.f, 25.f, 27.f,
        4.f, 6.f, 12.f, 14.f, 20.f, 22.f, 28.f, 30.f, 5.f, 7.f, 13.f, 15.f, 21.f, 23.f, 29.f, 31.f,
    });
    test_case.add_expected_output<float>(
        Shape{1, 2, 4, 4}, {0.f,  1.f,  2.f,  3.f,  4.f,  5.f,  6.f,  7.f,  8.f,  9.f,  10.f,
                            11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f,
                            22.f, 23.f, 24.f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 31.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_across_chw_4d)
{
    Shape data_shape{1, 2, 3, 4};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::Constant>(element::i64, Shape{3}, vector<int64_t>{1, 2, 3});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    auto normalize = make_shared<op::NormalizeL2>(data, axes, eps, eps_mode);
    auto function = make_shared<Function>(NodeVector{normalize}, ParameterVector{data});

    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    test_case.add_expected_output<float>(
        data_shape, {0.01428571f, 0.02857143f, 0.04285714f, 0.05714286f, 0.07142857f, 0.08571429f,
                     0.1f,        0.11428571f, 0.12857144f, 0.14285715f, 0.15714286f, 0.17142858f,
                     0.18571429f, 0.2f,        0.21428572f, 0.22857143f, 0.24285714f, 0.25714287f,
                     0.27142859f, 0.2857143f,  0.30000001f, 0.31428573f, 0.32857144f, 0.34285715f});

    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_across_empty_axes_input)
{
    Shape data_shape{1, 2, 3, 4};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::Constant>(element::i64, Shape{0}, vector<int64_t>{});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    auto normalize = make_shared<op::NormalizeL2>(data, axes, eps, eps_mode);
    auto function = make_shared<Function>(NodeVector{normalize}, ParameterVector{data});

    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    // output should be filled with 1f values
    test_case.add_expected_output<float>(data_shape, vector<float>(shape_size(data_shape), 1));

    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_across_h_4d)
{
    Shape data_shape{1, 2, 3, 4};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::Constant>(element::i64, Shape{1}, vector<int64_t>{1});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    auto normalize = make_shared<op::NormalizeL2>(data, axes, eps, eps_mode);
    auto function = make_shared<Function>(NodeVector{normalize}, ParameterVector{data});

    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    test_case.add_expected_output<float>(
        data_shape, {0.0766965f,  0.14142136f, 0.19611613f, 0.24253564f, 0.28216633f, 0.31622776f,
                     0.34570536f, 0.37139067f, 0.39391932f, 0.41380295f, 0.43145549f, 0.44721359f,
                     0.99705452f, 0.98994946f, 0.98058069f, 0.97014254f, 0.95936549f, 0.94868332f,
                     0.93834311f, 0.92847669f, 0.91914505f, 0.91036648f, 0.90213418f, 0.89442718f});
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_across_1axis_5d)
{
    Shape data_shape{1, 2, 2, 2, 3};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::Constant>(element::i64, Shape{1}, vector<int64_t>{1});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    auto normalize = make_shared<op::NormalizeL2>(data, axes, eps, eps_mode);
    auto function = make_shared<Function>(NodeVector{normalize}, ParameterVector{data});

    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    test_case.add_expected_output<float>(
        data_shape, {0.0766965f,  0.14142136f, 0.19611613f, 0.24253564f, 0.28216633f, 0.31622776f,
                     0.34570536f, 0.37139067f, 0.39391932f, 0.41380295f, 0.43145549f, 0.44721359f,
                     0.99705452f, 0.98994946f, 0.98058069f, 0.97014254f, 0.95936549f, 0.94868332f,
                     0.93834311f, 0.92847669f, 0.91914505f, 0.91036648f, 0.90213418f, 0.89442718f});
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_across_123axes_5d)
{
    Shape data_shape{1, 2, 2, 2, 3};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::Constant>(element::i64, Shape{3}, vector<int64_t>{1, 2, 3});
    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    auto normalize = make_shared<op::NormalizeL2>(data, axes, eps, eps_mode);
    auto function = make_shared<Function>(NodeVector{normalize}, ParameterVector{data});

    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    test_case.add_expected_output<float>(
        data_shape, {0.02638899f, 0.04956816f, 0.070014f,   0.10555596f, 0.1239204f,  0.140028f,
                     0.18472293f, 0.19827265f, 0.210042f,   0.26388991f, 0.27262488f, 0.280056f,
                     0.34305686f, 0.34697714f, 0.35007f,    0.42222384f, 0.42132938f, 0.420084f,
                     0.50139081f, 0.49568161f, 0.49009803f, 0.58055776f, 0.57003385f, 0.560112f});
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_across_chw_4d_max_bias)
{
    Shape data_shape{1, 2, 3, 4};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto axes = make_shared<op::Constant>(element::i64, Shape{3}, vector<int64_t>{1, 2, 3});
    float eps{5000};
    auto eps_mode = op::EpsMode::MAX;

    auto normalize = make_shared<op::NormalizeL2>(data, axes, eps, eps_mode);
    auto function = make_shared<Function>(NodeVector{normalize}, ParameterVector{data});

    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    test_case.add_expected_output<float>(
        data_shape, {0.01414214f, 0.02828427f, 0.04242641f, 0.05656854f, 0.07071068f, 0.08485281f,
                     0.09899495f, 0.11313709f, 0.12727922f, 0.14142136f, 0.15556349f, 0.16970563f,
                     0.18384777f, 0.1979899f,  0.21213204f, 0.22627418f, 0.2404163f,  0.25455844f,
                     0.26870057f, 0.28284273f, 0.29698485f, 0.31112698f, 0.32526913f, 0.33941126f});

    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

NGRAPH_TEST(${BACKEND_NAME}, gemm)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{3, 6});
    auto B = make_shared<op::Parameter>(element::f32, Shape{6, 4});
    auto C = make_shared<op::Parameter>(element::f32, Shape{3, 4});

    auto gemm_func = make_shared<op::Gemm>(A, B, C);
    auto function = make_shared<Function>(NodeVector{gemm_func}, ParameterVector{A, B, C});
    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");
    // A
    test_case.add_input<float>(vector<float>(18, 1));
    // B
    test_case.add_input<float>(vector<float>(24, 2));
    // C
    test_case.add_input<float>(vector<float>(12, 0));
    // output
    test_case.add_expected_output<float>(Shape{3, 4}, vector<float>(12, 12));
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, gemm_broadcast_input_C)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{3, 6});
    auto B = make_shared<op::Parameter>(element::f32, Shape{6, 4});
    auto C = make_shared<op::Parameter>(element::f32, Shape{});

    auto gemm_func = make_shared<op::Gemm>(A, B, C, 0.5);
    auto function = make_shared<Function>(NodeVector{gemm_func}, ParameterVector{A, B, C});
    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");
    // A
    test_case.add_input<float>(vector<float>(18, 1));
    // B
    test_case.add_input<float>(vector<float>(24, 2));
    // C
    test_case.add_input<float>(vector<float>{1});
    // output
    test_case.add_expected_output<float>(Shape{3, 4}, vector<float>(12, 7));
}

NGRAPH_TEST(${BACKEND_NAME}, fused_clamp)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{4, 4});
    auto tested_op = make_shared<op::Clamp>(data, 10.0, 20.0);
    auto function = make_shared<Function>(tested_op, ParameterVector{data});

    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({numeric_limits<float>::min(),
                                numeric_limits<float>::max(),
                                -numeric_limits<float>::infinity(),
                                numeric_limits<float>::infinity(),
                                -1.0,
                                0.0,
                                1.0,
                                9.99999,
                                10.0,
                                10.000001,
                                15.0,
                                19.999999,
                                20.0,
                                20.000001,
                                21.0,
                                100.0});

    test_case.add_expected_output<float>(Shape{4, 4},
                                         {10.0,
                                          20.0,
                                          10.0,
                                          20.0,
                                          10.0,
                                          10.0,
                                          10.0,
                                          10.0,
                                          10.0,
                                          10.000001,
                                          15.0,
                                          19.999999,
                                          20.0,
                                          20.0,
                                          20.0,
                                          20.0});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, mvn_mean_normalization)
{
    Shape data_shape{1, 2, 5};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);

    auto mvn_func = make_shared<op::MVN>(data, true, false);
    auto function = make_shared<Function>(NodeVector{mvn_func}, ParameterVector{data});
    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");
    // data
    vector<float> data_vector(shape_size(data_shape));
    iota(begin(data_vector), end(data_vector), 0);
    test_case.add_input<float>(data_vector);

    // expected result
    test_case.add_expected_output<float>(
        data_shape, vector<float>{-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, mvn_mean_normalization_split_channels)
{
    Shape data_shape{1, 2, 5, 1};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);

    auto mvn_func = make_shared<op::MVN>(data, false, false);
    auto function = make_shared<Function>(NodeVector{mvn_func}, ParameterVector{data});
    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");
    // data
    vector<float> data_vector(shape_size(data_shape));
    iota(begin(data_vector), end(data_vector), 0);
    test_case.add_input<float>(data_vector);

    // expected result
    test_case.add_expected_output<float>({1, 2, 5, 1},
                                         vector<float>{-2, -1, 0, 1, 2, -2, -1, 0, 1, 2});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, mvn_mean_variance_normalization)
{
    Shape data_shape{1, 2, 5};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);

    auto mvn_func = make_shared<op::MVN>(data);
    auto function = make_shared<Function>(NodeVector{mvn_func}, ParameterVector{data});
    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");
    // data
    vector<float> data_vector(shape_size(data_shape));
    iota(begin(data_vector), end(data_vector), 0);
    test_case.add_input<float>(data_vector);

    // expected result
    test_case.add_expected_output<float>(data_shape,
                                         vector<float>{-1.566698903055826,
                                                       -1.2185435912656424,
                                                       -0.87038827947545883,
                                                       -0.52223296768527527,
                                                       -0.17407765589509178,
                                                       0.17407765589509178,
                                                       0.52223296768527527,
                                                       0.87038827947545883,
                                                       1.2185435912656424,
                                                       1.566698903055826});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, mvn_mean_variance_normalization_split_channels)
{
    Shape data_shape{1, 2, 5};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);

    auto mvn_func = make_shared<op::MVN>(data, false);
    auto function = make_shared<Function>(NodeVector{mvn_func}, ParameterVector{data});
    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");
    // data
    vector<float> data_vector(shape_size(data_shape));
    iota(begin(data_vector), end(data_vector), 0);
    test_case.add_input<float>(data_vector);

    // expected result
    test_case.add_expected_output<float>(data_shape,
                                         vector<float>{-1.4142135613730948,
                                                       -0.70710678068654742,
                                                       0.000000000000000,
                                                       0.70710678068654742,
                                                       1.4142135613730948,
                                                       -1.4142135613730948,
                                                       -0.70710678068654742,
                                                       0.000000000000000,
                                                       0.70710678068654742,
                                                       1.4142135613730948});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, grn_4d)
{
    const Shape data_shape{1, 2, 3, 4};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);
    float bias{1e-6f};

    const auto grn = make_shared<op::GRN>(data, bias);
    const auto function = make_shared<Function>(NodeVector{grn}, ParameterVector{data});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    test_case.add_expected_output<float>(
        data_shape, {0.0766965f,  0.14142136f, 0.19611613f, 0.24253564f, 0.28216633f, 0.31622776f,
                     0.34570536f, 0.37139067f, 0.39391932f, 0.41380295f, 0.4314555f,  0.4472136f,
                     0.9970545f,  0.98994946f, 0.9805807f,  0.97014254f, 0.9593655f,  0.9486833f,
                     0.9383431f,  0.9284767f,  0.91914505f, 0.9103665f,  0.9021342f,  0.8944272f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, grn_2d_with_bias)
{
    const Shape data_shape{3, 4};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);
    float bias{2.25f};

    const auto grn = make_shared<op::GRN>(data, bias);
    const auto function = make_shared<Function>(NodeVector{grn}, ParameterVector{data});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);

    test_case.add_expected_output<float>(data_shape,
                                         {0.5547002f,
                                          0.8f,
                                          0.8944272f,
                                          0.9363292f,
                                          0.95782626f,
                                          0.9701425f,
                                          0.9778024f,
                                          0.98287225f,
                                          0.9863939f,
                                          0.9889363f,
                                          0.9908301f,
                                          0.99227786f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, unsqueeze)
{
    auto data_node = make_shared<op::Parameter>(element::f32, Shape{4, 2});
    auto axes_node =
        make_shared<ngraph::op::Constant>(element::u64, Shape{2}, vector<int64_t>{1, 2});
    auto squeeze = make_shared<op::Unsqueeze>(data_node, axes_node);

    auto function = make_shared<Function>(NodeVector{squeeze}, ParameterVector{data_node});
    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    auto data = vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    test_case.add_input(data);
    test_case.add_expected_output<float>(Shape{4, 1, 1, 2}, data);
}

NGRAPH_TEST(${BACKEND_NAME}, scale_shift_no_broadcast)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{3, 6});
    auto scale = make_shared<op::Parameter>(element::f32, Shape{3, 6});
    auto shift = make_shared<op::Parameter>(element::f32, Shape{3, 6});

    auto scale_shift_func = make_shared<op::ScaleShift>(data, scale, shift);
    auto function =
        make_shared<Function>(NodeVector{scale_shift_func}, ParameterVector{data, scale, shift});
    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");
    // Data
    test_case.add_input<float>(vector<float>(18, 2));
    // Scale
    test_case.add_input<float>(vector<float>(18, 2));
    // Shift
    test_case.add_input<float>(vector<float>(18, 2));
    // output
    test_case.add_expected_output<float>(Shape{3, 6}, vector<float>(18, 6));
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, scale_shift)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{3, 6});
    auto scale = make_shared<op::Parameter>(element::f32, Shape{3, 6});
    auto shift = make_shared<op::Parameter>(element::f32, Shape{});

    auto scale_shift_func = make_shared<op::ScaleShift>(data, scale, shift);
    auto function =
        make_shared<Function>(NodeVector{scale_shift_func}, ParameterVector{data, scale, shift});
    auto test_case = test::NgraphTestCase(function, "${BACKEND_NAME}");
    // Data
    test_case.add_input<float>(vector<float>(18, 2));
    // Scale
    test_case.add_input<float>(vector<float>(18, 2));
    // Shift
    test_case.add_input<float>(vector<float>{2});
    // output
    test_case.add_expected_output<float>(Shape{3, 6}, vector<float>(18, 6));
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, shuffle_channels_simple)
{
    const auto data = make_shared<op::Parameter>(element::i32, Shape{1, 15, 2, 2});
    auto tested_op = make_shared<op::ShuffleChannels>(data, 1, 5);
    auto function = make_shared<Function>(tested_op, ParameterVector{data});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    std::vector<int32_t> input_data(60);
    std::iota(std::begin(input_data), std::end(input_data), 0);
    test_case.add_input(input_data);

    test_case.add_expected_output<int32_t>(
        Shape{1, 15, 2, 2},
        {0, 1, 2,  3,  12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39, 48, 49, 50, 51,
         4, 5, 6,  7,  16, 17, 18, 19, 28, 29, 30, 31, 40, 41, 42, 43, 52, 53, 54, 55,
         8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35, 44, 45, 46, 47, 56, 57, 58, 59});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, shuffle_channels_negative_axis)
{
    // in this test the output is the same as in shuffle_channels_simple but
    // the axis value is negative and the C(channels) value is in a different dimension(0) of the
    // shape
    const auto data = make_shared<op::Parameter>(element::i32, Shape{15, 2, 1, 2});
    auto tested_op = make_shared<op::ShuffleChannels>(data, -4, 5);
    auto function = make_shared<Function>(tested_op, ParameterVector{data});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    std::vector<int32_t> input_data(60);
    std::iota(std::begin(input_data), std::end(input_data), 0);
    test_case.add_input(input_data);

    test_case.add_expected_output<int32_t>(
        Shape{15, 2, 1, 2},
        {0, 1, 2,  3,  12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39, 48, 49, 50, 51,
         4, 5, 6,  7,  16, 17, 18, 19, 28, 29, 30, 31, 40, 41, 42, 43, 52, 53, 54, 55,
         8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35, 44, 45, 46, 47, 56, 57, 58, 59});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, shuffle_channels_float)
{
    const auto data = make_shared<op::Parameter>(element::f32, Shape{6, 1, 1, 1});
    auto tested_op = make_shared<op::ShuffleChannels>(data, 0, 2);
    auto function = make_shared<Function>(tested_op, ParameterVector{data});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    test_case.add_input<float>({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    test_case.add_expected_output<float>(Shape{6, 1, 1, 1}, {0.0f, 3.0f, 1.0f, 4.0f, 2.0f, 5.0f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, squeeze)
{
    const auto data_node = make_shared<op::Parameter>(element::f32, Shape{1, 4, 1, 1, 2});
    const auto axes_node =
        make_shared<ngraph::op::Constant>(element::u64, Shape{2}, vector<int64_t>{0, 2});
    const auto squeeze = make_shared<op::Squeeze>(data_node, axes_node);

    const auto function = make_shared<Function>(NodeVector{squeeze}, ParameterVector{data_node});
    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    const auto data = vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    test_case.add_input(data);
    test_case.add_expected_output<float>(Shape{4, 1, 2}, data);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, squeeze_default_axes)
{
    const auto data_node = make_shared<op::Parameter>(element::f32, Shape{1, 4, 1, 1, 2});
    const auto axes_node =
        make_shared<ngraph::op::Constant>(element::u64, Shape{0}, vector<int64_t>{});
    const auto squeeze = make_shared<op::Squeeze>(data_node, axes_node);

    const auto function = make_shared<Function>(NodeVector{squeeze}, ParameterVector{data_node});
    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    const auto data = vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    test_case.add_input(data);
    test_case.add_expected_output<float>(Shape{4, 2}, data);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, squeeze_dynamic)
{
    const auto data_param = make_shared<op::Parameter>(element::f32, Shape{1, 4, 1, 1, 2});
    const auto axes_param = make_shared<op::Parameter>(element::i64, Shape{2});
    EXPECT_THROW(make_shared<op::Squeeze>(data_param, axes_param), CheckFailure);
}

NGRAPH_TEST(${BACKEND_NAME}, squared_difference)
{
    const auto x1 = make_shared<op::Parameter>(element::f32, Shape{2, 2});
    const auto x2 = make_shared<op::Parameter>(element::f32, Shape{2, 2});

    auto tested_op = make_shared<op::SquaredDifference>(x1, x2);
    auto function = make_shared<Function>(tested_op, ParameterVector{x1, x2});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<float>({1.0, 16.0, 0.0, 1.234567});
    test_case.add_input<float>({1.0, 8.0, -3.0, 3.456789});

    test_case.add_expected_output<float>(Shape{2, 2}, {0.0, 64.0, 9.0, 4.938270617284});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, squared_difference_broadcast)
{
    const auto x1 = make_shared<op::Parameter>(element::i32, Shape{2, 2});
    const auto x2 = make_shared<op::Parameter>(element::i32, Shape{});

    auto tested_op = make_shared<op::SquaredDifference>(x1, x2);
    auto function = make_shared<Function>(tested_op, ParameterVector{x1, x2});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<int32_t>({1, 1, 1, 1});
    test_case.add_input<int32_t>({1});

    test_case.add_expected_output<int32_t>(Shape{2, 2}, {0, 0, 0, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, split_3_equal_parts)
{
    const auto data = make_shared<op::Parameter>(element::i32, Shape{6});

    const auto tested_op = make_shared<op::Split>(data, 0, 3);
    const auto function = make_shared<Function>(tested_op->decompose_op(), ParameterVector{data});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<int32_t>({1, 2, 3, 4, 5, 6});

    test_case.add_expected_output<int32_t>(Shape{2}, {1, 2});
    test_case.add_expected_output<int32_t>(Shape{2}, {3, 4});
    test_case.add_expected_output<int32_t>(Shape{2}, {5, 6});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, split_var_len_parts)
{
    const auto data = make_shared<op::Parameter>(element::i32, Shape{2, 6});

    const std::vector<size_t> splits = {2, 4};
    const auto tested_op = make_shared<op::Split>(data, 1, splits);
    const auto function = make_shared<Function>(tested_op->decompose_op(), ParameterVector{data});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<int32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});

    test_case.add_expected_output<int32_t>(Shape{2, 2}, {0, 1, 6, 7});
    test_case.add_expected_output<int32_t>(Shape{2, 4}, {2, 3, 4, 5, 8, 9, 10, 11});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, lstm_cell_no_bias_no_peepholes)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto C_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});

    const auto lstm_cell = make_shared<op::LSTMCell>(X, W, R, H_t, C_t, hidden_size);

    auto ht_function = make_shared<Function>(make_shared<op::GetOutputElement>(lstm_cell, 0),
                                             ParameterVector{X, W, R, H_t, C_t});
    auto ht_test_case = ngraph::test::NgraphTestCase(ht_function, "${BACKEND_NAME}");
    // X
    vector<float> in_X{0.81342685f, 0.84108883f, 0.8152282f, 0.46893653f, 0.0901856f, 0.37088776f};
    // W
    vector<float> in_W{3.3330739e-01f, 3.6229487e-04f, 4.6773660e-01f, 4.3046016e-01f,
                       7.3950343e-02f, 3.8063636e-01f, 9.6921772e-01f, 9.6897459e-01f,
                       6.2964785e-01f, 3.1134409e-01f, 8.4709978e-01f, 9.4928098e-01f,
                       6.1676943e-01f, 6.6020679e-01f, 1.9072217e-01f, 8.8032126e-02f,
                       4.0472135e-01f, 6.8342745e-01f, 8.3432144e-01f, 4.4928190e-01f,
                       7.9524308e-01f, 5.3966165e-01f, 8.5936421e-01f, 8.3136767e-01f,
                       5.5125546e-02f, 4.7791195e-01f, 3.5788772e-01f, 6.7507404e-01f,
                       2.1716513e-01f, 2.7473119e-01f, 3.3999152e-02f, 9.6835363e-01f,
                       3.7581277e-01f, 2.4026000e-01f, 6.7418844e-01f, 3.4199652e-01f};
    // R
    vector<float> in_R{
        0.0987983f,  0.52032113f, 0.5848073f,  0.5356095f,  0.74497133f, 0.73260087f,
        0.1700787f,  0.45684233f, 0.1495722f,  0.42734373f, 0.4433832f,  0.25906256f,
        0.03854987f, 0.47480518f, 0.37215272f, 0.99890584f, 0.74019486f, 0.3518967f,
        0.6881257f,  0.8170279f,  0.54088944f, 0.81225616f, 0.14619833f, 0.42941234f,
        0.86843914f, 0.45967972f, 0.6237719f,  0.11074839f, 0.6029616f,  0.3149305f,
        0.46504205f, 0.5843412f,  0.8733427f,  0.7687243f,  0.07074859f, 0.39188156f};
    // Ht
    vector<float> in_Ht{0.77956f, 0.5331557f, 0.04297554f, 0.7962175f, 0.7635707f, 0.11989366f};
    // Ct
    vector<float> in_Ct{0.8488452f, 0.18851636f, 0.5020695f, 0.29716516f, 0.06740791f, 0.45384037f};

    ht_test_case.add_multiple_inputs(vector<vector<float>>{in_X, in_W, in_R, in_Ht, in_Ct});
    ht_test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.81457126f, 0.61109227f, 0.769522f, 0.52239674f, 0.4324641f, 0.63183f});
    ht_test_case.run();

    auto ct_function = make_shared<Function>(make_shared<op::GetOutputElement>(lstm_cell, 1),
                                             ParameterVector{X, W, R, H_t, C_t});
    auto ct_test_case = ngraph::test::NgraphTestCase(ct_function, "${BACKEND_NAME}");
    ct_test_case.add_multiple_inputs(vector<vector<float>>{in_X, in_W, in_R, in_Ht, in_Ct});
    ct_test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {1.4444952f, 0.9635685f, 1.2875274f, 0.8053419f, 0.7184521f, 0.95803297f});
    ct_test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, lstm_cell_bias_peepholes)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto C_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{2 * gates_count * hidden_size});
    const auto P = make_shared<op::Parameter>(element::f32, Shape{3 * hidden_size});

    const auto lstm_cell = make_shared<op::LSTMCell>(X, W, R, H_t, C_t, hidden_size, B, P);

    auto ht_function = make_shared<Function>(make_shared<op::GetOutputElement>(lstm_cell, 0),
                                             ParameterVector{X, W, R, H_t, C_t, B, P});
    auto ht_test_case = ngraph::test::NgraphTestCase(ht_function, "${BACKEND_NAME}");

    // X
    vector<float> in_X{0.81342685f, 0.84108883f, 0.8152282f, 0.46893653f, 0.0901856f, 0.37088776f};
    // W
    vector<float> in_W{3.3330739e-01f, 3.6229487e-04f, 4.6773660e-01f, 4.3046016e-01f,
                       7.3950343e-02f, 3.8063636e-01f, 9.6921772e-01f, 9.6897459e-01f,
                       6.2964785e-01f, 3.1134409e-01f, 8.4709978e-01f, 9.4928098e-01f,
                       6.1676943e-01f, 6.6020679e-01f, 1.9072217e-01f, 8.8032126e-02f,
                       4.0472135e-01f, 6.8342745e-01f, 8.3432144e-01f, 4.4928190e-01f,
                       7.9524308e-01f, 5.3966165e-01f, 8.5936421e-01f, 8.3136767e-01f,
                       5.5125546e-02f, 4.7791195e-01f, 3.5788772e-01f, 6.7507404e-01f,
                       2.1716513e-01f, 2.7473119e-01f, 3.3999152e-02f, 9.6835363e-01f,
                       3.7581277e-01f, 2.4026000e-01f, 6.7418844e-01f, 3.4199652e-01f};
    // R
    vector<float> in_R{
        0.0987983f,  0.52032113f, 0.5848073f,  0.5356095f,  0.74497133f, 0.73260087f,
        0.1700787f,  0.45684233f, 0.1495722f,  0.42734373f, 0.4433832f,  0.25906256f,
        0.03854987f, 0.47480518f, 0.37215272f, 0.99890584f, 0.74019486f, 0.3518967f,
        0.6881257f,  0.8170279f,  0.54088944f, 0.81225616f, 0.14619833f, 0.42941234f,
        0.86843914f, 0.45967972f, 0.6237719f,  0.11074839f, 0.6029616f,  0.3149305f,
        0.46504205f, 0.5843412f,  0.8733427f,  0.7687243f,  0.07074859f, 0.39188156f};
    // Ht
    vector<float> in_Ht{0.77956f, 0.5331557f, 0.04297554f, 0.7962175f, 0.7635707f, 0.11989366f};
    // Ct
    vector<float> in_Ct{0.8488452f, 0.18851636f, 0.5020695f, 0.29716516f, 0.06740791f, 0.45384037f};
    // B
    vector<float> in_B{0.81130236f, 0.31332242f, 0.6423671f,  0.09981899f, 0.7847627f,
                       0.8405669f,  0.0330242f,  0.45014873f, 0.5599519f,  0.31807426f,
                       0.7356558f,  0.6298691f,  0.26263478f, 0.8391581f,  0.52434635f,
                       0.11468413f, 0.4533051f,  0.67632145f, 0.43415946f, 0.46795473f,
                       0.5674715f,  0.19214648f, 0.37824264f, 0.11187395f};
    // P
    vector<float> in_P{0.38557124f,
                       0.9482306f,
                       0.6808912f,
                       0.93585867f,
                       0.74540526f,
                       0.10507805f,
                       0.8180733f,
                       0.13840231f,
                       0.24175227f};

    ht_test_case.add_multiple_inputs(
        vector<vector<float>>{in_X, in_W, in_R, in_Ht, in_Ct, in_B, in_P});
    ht_test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.9218244f, 0.78787273f, 0.8754273f, 0.7361462f, 0.70927656f, 0.83522964f});
    ht_test_case.run();

    auto ct_function = make_shared<Function>(make_shared<op::GetOutputElement>(lstm_cell, 1),
                                             ParameterVector{X, W, R, H_t, C_t, B, P});
    auto ct_test_case = ngraph::test::NgraphTestCase(ct_function, "${BACKEND_NAME}");
    ct_test_case.add_multiple_inputs(
        vector<vector<float>>{in_X, in_W, in_R, in_Ht, in_Ct, in_B, in_P});
    ct_test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {1.7094649f, 1.1259761f, 1.444019f, 1.086587f, 0.9762144f, 1.3066899f});
    ct_test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, lstm_cell_bias_peepholes_clip_input_forget)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;
    const float clip_threshold = 3.5f;
    bool input_forget = true;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto C_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{2 * gates_count * hidden_size});
    const auto P = make_shared<op::Parameter>(element::f32, Shape{3 * hidden_size});

    const auto lstm_cell = make_shared<op::LSTMCell>(X,
                                                     W,
                                                     R,
                                                     H_t,
                                                     C_t,
                                                     hidden_size,
                                                     B,
                                                     P,
                                                     vector<string>{"sigmoid", "tanh", "tanh"},
                                                     vector<float>{},
                                                     vector<float>{},
                                                     clip_threshold,
                                                     input_forget);
    auto ht_function = make_shared<Function>(make_shared<op::GetOutputElement>(lstm_cell, 0),
                                             ParameterVector{X, W, R, H_t, C_t, B, P});
    auto ht_test_case = ngraph::test::NgraphTestCase(ht_function, "${BACKEND_NAME}");

    // X
    vector<float> in_X{0.81342685f, 0.84108883f, 0.8152282f, 0.46893653f, 0.0901856f, 0.37088776f};
    // W
    vector<float> in_W{3.3330739e-01f, 3.6229487e-04f, 4.6773660e-01f, 4.3046016e-01f,
                       7.3950343e-02f, 3.8063636e-01f, 9.6921772e-01f, 9.6897459e-01f,
                       6.2964785e-01f, 3.1134409e-01f, 8.4709978e-01f, 9.4928098e-01f,
                       6.1676943e-01f, 6.6020679e-01f, 1.9072217e-01f, 8.8032126e-02f,
                       4.0472135e-01f, 6.8342745e-01f, 8.3432144e-01f, 4.4928190e-01f,
                       7.9524308e-01f, 5.3966165e-01f, 8.5936421e-01f, 8.3136767e-01f,
                       5.5125546e-02f, 4.7791195e-01f, 3.5788772e-01f, 6.7507404e-01f,
                       2.1716513e-01f, 2.7473119e-01f, 3.3999152e-02f, 9.6835363e-01f,
                       3.7581277e-01f, 2.4026000e-01f, 6.7418844e-01f, 3.4199652e-01f};
    // R
    vector<float> in_R{
        0.0987983f,  0.52032113f, 0.5848073f,  0.5356095f,  0.74497133f, 0.73260087f,
        0.1700787f,  0.45684233f, 0.1495722f,  0.42734373f, 0.4433832f,  0.25906256f,
        0.03854987f, 0.47480518f, 0.37215272f, 0.99890584f, 0.74019486f, 0.3518967f,
        0.6881257f,  0.8170279f,  0.54088944f, 0.81225616f, 0.14619833f, 0.42941234f,
        0.86843914f, 0.45967972f, 0.6237719f,  0.11074839f, 0.6029616f,  0.3149305f,
        0.46504205f, 0.5843412f,  0.8733427f,  0.7687243f,  0.07074859f, 0.39188156f};
    // Ht
    vector<float> in_Ht{0.77956f, 0.5331557f, 0.04297554f, 0.7962175f, 0.7635707f, 0.11989366f};
    // Ct
    vector<float> in_Ct{0.8488452f, 0.18851636f, 0.5020695f, 0.29716516f, 0.06740791f, 0.45384037f};
    // B
    vector<float> in_B{0.81130236f, 0.31332242f, 0.6423671f,  0.09981899f, 0.7847627f,
                       0.8405669f,  0.0330242f,  0.45014873f, 0.5599519f,  0.31807426f,
                       0.7356558f,  0.6298691f,  0.26263478f, 0.8391581f,  0.52434635f,
                       0.11468413f, 0.4533051f,  0.67632145f, 0.43415946f, 0.46795473f,
                       0.5674715f,  0.19214648f, 0.37824264f, 0.11187395f};
    // P
    vector<float> in_P{0.38557124f,
                       0.9482306f,
                       0.6808912f,
                       0.93585867f,
                       0.74540526f,
                       0.10507805f,
                       0.8180733f,
                       0.13840231f,
                       0.24175227f};

    ht_test_case.add_multiple_inputs(
        vector<vector<float>>{in_X, in_W, in_R, in_Ht, in_Ct, in_B, in_P});
    ht_test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.71485436f, 0.71844107f, 0.72704613f, 0.6235602f, 0.68306124f, 0.6978715f});
    ht_test_case.run();

    auto ct_function = make_shared<Function>(make_shared<op::GetOutputElement>(lstm_cell, 1),
                                             ParameterVector{X, W, R, H_t, C_t, B, P});
    auto ct_test_case = ngraph::test::NgraphTestCase(ct_function, "${BACKEND_NAME}");
    ct_test_case.add_multiple_inputs(
        vector<vector<float>>{in_X, in_W, in_R, in_Ht, in_Ct, in_B, in_P});
    ct_test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.94656503f, 0.9527454f, 0.9706756f, 0.84206575f, 0.91898793f, 0.9127192f});
    ct_test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, lstm_cell_activaction_functions)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 4;
    const float clip_threshold = 3.5f;
    bool input_forget = true;
    vector<string> activations{"sigmoid", "tanh", "hardsigmoid"};
    vector<float> activation_alpha{0.f, 0.f, 1.8345f};
    vector<float> activation_beta{0.f, 0.f, 3.05f};

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto C_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{2 * gates_count * hidden_size});
    const auto P = make_shared<op::Parameter>(element::f32, Shape{3 * hidden_size});

    const auto lstm_cell = make_shared<op::LSTMCell>(X,
                                                     W,
                                                     R,
                                                     H_t,
                                                     C_t,
                                                     hidden_size,
                                                     B,
                                                     P,
                                                     activations,
                                                     activation_alpha,
                                                     activation_beta,
                                                     clip_threshold,
                                                     input_forget);
    auto ht_function = make_shared<Function>(make_shared<op::GetOutputElement>(lstm_cell, 0),
                                             ParameterVector{X, W, R, H_t, C_t, B, P});
    auto ht_test_case = ngraph::test::NgraphTestCase(ht_function, "${BACKEND_NAME}");

    // X
    vector<float> in_X{0.81342685f, 0.84108883f, 0.8152282f, 0.46893653f, 0.0901856f, 0.37088776f};
    // W
    vector<float> in_W{3.3330739e-01f, 3.6229487e-04f, 4.6773660e-01f, 4.3046016e-01f,
                       7.3950343e-02f, 3.8063636e-01f, 9.6921772e-01f, 9.6897459e-01f,
                       6.2964785e-01f, 3.1134409e-01f, 8.4709978e-01f, 9.4928098e-01f,
                       6.1676943e-01f, 6.6020679e-01f, 1.9072217e-01f, 8.8032126e-02f,
                       4.0472135e-01f, 6.8342745e-01f, 8.3432144e-01f, 4.4928190e-01f,
                       7.9524308e-01f, 5.3966165e-01f, 8.5936421e-01f, 8.3136767e-01f,
                       5.5125546e-02f, 4.7791195e-01f, 3.5788772e-01f, 6.7507404e-01f,
                       2.1716513e-01f, 2.7473119e-01f, 3.3999152e-02f, 9.6835363e-01f,
                       3.7581277e-01f, 2.4026000e-01f, 6.7418844e-01f, 3.4199652e-01f};
    // R
    vector<float> in_R{
        0.0987983f,  0.52032113f, 0.5848073f,  0.5356095f,  0.74497133f, 0.73260087f,
        0.1700787f,  0.45684233f, 0.1495722f,  0.42734373f, 0.4433832f,  0.25906256f,
        0.03854987f, 0.47480518f, 0.37215272f, 0.99890584f, 0.74019486f, 0.3518967f,
        0.6881257f,  0.8170279f,  0.54088944f, 0.81225616f, 0.14619833f, 0.42941234f,
        0.86843914f, 0.45967972f, 0.6237719f,  0.11074839f, 0.6029616f,  0.3149305f,
        0.46504205f, 0.5843412f,  0.8733427f,  0.7687243f,  0.07074859f, 0.39188156f};
    // Ht
    vector<float> in_Ht{0.77956f, 0.5331557f, 0.04297554f, 0.7962175f, 0.7635707f, 0.11989366f};
    // Ct
    vector<float> in_Ct{0.8488452f, 0.18851636f, 0.5020695f, 0.29716516f, 0.06740791f, 0.45384037f};
    // B
    vector<float> in_B{0.81130236f, 0.31332242f, 0.6423671f,  0.09981899f, 0.7847627f,
                       0.8405669f,  0.0330242f,  0.45014873f, 0.5599519f,  0.31807426f,
                       0.7356558f,  0.6298691f,  0.26263478f, 0.8391581f,  0.52434635f,
                       0.11468413f, 0.4533051f,  0.67632145f, 0.43415946f, 0.46795473f,
                       0.5674715f,  0.19214648f, 0.37824264f, 0.11187395f};
    // P
    vector<float> in_P{0.38557124f,
                       0.9482306f,
                       0.6808912f,
                       0.93585867f,
                       0.74540526f,
                       0.10507805f,
                       0.8180733f,
                       0.13840231f,
                       0.24175227f};

    ht_test_case.add_multiple_inputs(
        vector<vector<float>>{in_X, in_W, in_R, in_Ht, in_Ct, in_B, in_P});
    ht_test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.96834344f, 0.9695254f, 0.97068775f, 0.9077866f, 0.94161016f, 0.96599925f});
    ht_test_case.run();

    auto ct_function = make_shared<Function>(make_shared<op::GetOutputElement>(lstm_cell, 1),
                                             ParameterVector{X, W, R, H_t, C_t, B, P});
    auto ct_test_case = ngraph::test::NgraphTestCase(ct_function, "${BACKEND_NAME}");
    ct_test_case.add_multiple_inputs(
        vector<vector<float>>{in_X, in_W, in_R, in_Ht, in_Ct, in_B, in_P});
    ct_test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.94656503f, 0.9527454f, 0.9706756f, 0.84206575f, 0.91898793f, 0.9127192f});
    ct_test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, fake_quantize)
{
    const Shape data_shape{1, 2, 3, 4};
    const size_t levels = 4;
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto input_low = make_shared<op::Parameter>(element::f32, Shape{});
    const auto input_high = make_shared<op::Parameter>(element::f32, Shape{});
    const auto output_low = make_shared<op::Parameter>(element::f32, Shape{});
    const auto output_high = make_shared<op::Parameter>(element::f32, Shape{});

    const auto quantize =
        make_shared<op::FakeQuantize>(data, input_low, input_high, output_low, output_high, levels);
    const auto function = make_shared<Function>(
        NodeVector{quantize},
        ParameterVector{data, input_low, input_high, output_low, output_high});
    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    const size_t n_elements = shape_size(data_shape);
    vector<float> input_data(n_elements);
    iota(begin(input_data), end(input_data), 0);

    test_case.add_input<float>(input_data);
    // input_low
    test_case.add_input<float>({0.0f});
    // input_high
    test_case.add_input<float>({23.f});
    // output_low
    test_case.add_input<float>({2.f});
    // output_high
    test_case.add_input<float>({16.f});

    // expected result
    test_case.add_expected_output<float>(
        data_shape,
        vector<float>{2.f,          2.f,          2.f,          2.f,          6.6666669f,
                      6.6666669f,   6.6666669f,   6.6666669f,   6.6666669f,   6.6666669f,
                      6.6666669f,   6.6666669f,   11.33333301f, 11.33333301f, 11.33333301f,
                      11.33333301f, 11.33333301f, 11.33333301f, 11.33333301f, 11.33333301f,
                      16.f,         16.f,         16.f,         16.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, fake_quantize_with_clip)
{
    const Shape data_shape{1, 2, 3, 4};
    const size_t levels = 5;
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);
    const auto input_low = make_shared<op::Parameter>(element::f32, Shape{});
    const auto input_high = make_shared<op::Parameter>(element::f32, Shape{});
    const auto output_low = make_shared<op::Parameter>(element::f32, Shape{});
    const auto output_high = make_shared<op::Parameter>(element::f32, Shape{});

    const auto quantize =
        make_shared<op::FakeQuantize>(data, input_low, input_high, output_low, output_high, levels);
    const auto function = make_shared<Function>(
        NodeVector{quantize},
        ParameterVector{data, input_low, input_high, output_low, output_high});
    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    const size_t n_elements = shape_size(data_shape);
    vector<float> input_data(n_elements);
    iota(begin(input_data), end(input_data), 0);

    test_case.add_input<float>(input_data);
    // input_low
    test_case.add_input<float>({3.f});
    // input_high
    test_case.add_input<float>({17.f});
    // output_low
    test_case.add_input<float>({2.f});
    // output_high
    test_case.add_input<float>({16.f});

    // expected result
    test_case.add_expected_output<float>(
        data_shape,
        vector<float>{2.f,   2.f,   2.f,   2.f,   2.f,  5.5f, 5.5f, 5.5f, 5.5f, 9.f,  9.f,  9.f,
                      12.5f, 12.5f, 12.5f, 12.5f, 16.f, 16.f, 16.f, 16.f, 16.f, 16.f, 16.f, 16.f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, fake_quantize_with_clip_across_channels)
{
    Shape data_shape{1, 2, 5, 5};
    size_t levels = 5;
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto input_low = make_shared<op::Parameter>(element::f32, Shape{2});
    auto input_high = make_shared<op::Parameter>(element::f32, Shape{2});
    auto output_low = make_shared<op::Parameter>(element::f32, Shape{2});
    auto output_high = make_shared<op::Parameter>(element::f32, Shape{2});

    auto quantize =
        make_shared<op::FakeQuantize>(data, input_low, input_high, output_low, output_high, levels);
    auto function = make_shared<Function>(
        NodeVector{quantize},
        ParameterVector{data, input_low, input_high, output_low, output_high});
    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    size_t n_elements = shape_size(data_shape);
    vector<float> input_data(n_elements);
    iota(begin(input_data), end(input_data), 0);

    test_case.add_input<float>(input_data);
    // input_low
    test_case.add_input<float>(vector<float>{5.f, 30.f});
    // input_high
    test_case.add_input<float>(vector<float>{10.f, 40.f});
    // output_low
    test_case.add_input<float>(vector<float>{0.f, 50.f});
    // output_high
    test_case.add_input<float>(vector<float>{20.f, 70.f});

    // expected result
    test_case.add_expected_output<float>(
        data_shape,
        vector<float>{0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  5.0f,  10.0f, 10.0f, 15.0f,
                      20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 20.0f,
                      20.0f, 20.0f, 20.0f, 20.0f, 20.0f, 50.0f, 50.0f, 50.0f, 50.0f, 50.0f,
                      50.0f, 50.0f, 55.0f, 55.0f, 60.0f, 60.0f, 60.0f, 65.0f, 65.0f, 70.0f,
                      70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f, 70.0f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, group_conv_transpose)
{
    const CoordinateDiff output_padding{1, 1};
    const CoordinateDiff padding_begin{1, 1};
    const CoordinateDiff padding_end{1, 1};
    Strides strides{2, 2};
    Strides dilations{1, 1};
    size_t groups = 1;

    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 1, 3, 3});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{1, 1, 3, 3});

    auto gct = make_shared<op::GroupConvolutionTranspose>(
        data, filters, strides, dilations, padding_begin, padding_end, output_padding, groups);

    auto function = make_shared<Function>(NodeVector{gct}, ParameterVector{data, filters});
    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    // X
    test_case.add_input<float>(vector<float>{0.16857791f,
                                             -0.15161794f,
                                             0.08540368f,
                                             0.1820628f,
                                             -0.21746576f,
                                             0.08245695f,
                                             0.1431433f,
                                             -0.43156421f,
                                             0.30591947f});
    // W
    test_case.add_input<float>({-0.06230065f,
                                0.37932432f,
                                -0.25388849f,
                                0.33878803f,
                                0.43709868f,
                                -0.22477469f,
                                0.04118127f,
                                -0.44696793f,
                                0.06373066f});
    test_case.add_expected_output(
        Shape{1, 1, 6, 6},
        vector<float>{
            0.07368518f,  -0.08925839f, -0.06627201f, 0.06301362f,  0.03732984f,  -0.01919658f,
            -0.00628807f, -0.02817563f, -0.01472169f, 0.04392925f,  -0.00689478f, -0.01549204f,
            0.07957941f,  -0.11459791f, -0.09505399f, 0.07681622f,  0.03604182f,  -0.01853423f,
            -0.0270785f,  -0.00680824f, -0.06650258f, 0.08004665f,  0.07918708f,  -0.0724144f,
            0.06256775f,  -0.17838378f, -0.18863615f, 0.20064656f,  0.133717f,    -0.06876295f,
            -0.06398046f, -0.00864975f, 0.19289537f,  -0.01490572f, -0.13673618f, 0.01949645f});
    test_case.run(DEFAULT_FLOAT_TOLERANCE_BITS + 1);
}

NGRAPH_TEST(${BACKEND_NAME}, group_conv_transpose_output_shape)
{
    const CoordinateDiff output_padding{};
    const Shape output_shape{1, 1, 1, 14};
    Strides strides{1, 1};
    Strides dilations{1, 1};
    size_t groups = 1;

    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1, 10});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1, 5});

    auto gct = make_shared<op::GroupConvolutionTranspose>(
        data, filters, strides, dilations, output_padding, output_shape, groups);

    auto function = make_shared<Function>(NodeVector{gct}, ParameterVector{data, filters});
    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    // X
    test_case.add_input<float>(
        vector<float>{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    // W
    test_case.add_input<float>({1.0f, 2.0f, 3.0f, 2.0f, 1.0f});
    test_case.add_expected_output(Shape{1, 1, 1, 14},
                                  vector<float>{0.0f,
                                                1.0f,
                                                4.0f,
                                                10.0f,
                                                18.0f,
                                                27.0f,
                                                36.0f,
                                                45.0f,
                                                54.0f,
                                                63.0f,
                                                62.0f,
                                                50.0f,
                                                26.0f,
                                                9.0f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, rnn_cell_no_bias)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W = make_shared<op::Parameter>(element::f32, Shape{hidden_size, input_size});
    const auto R = make_shared<op::Parameter>(element::f32, Shape{hidden_size, hidden_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});

    const auto rnn_cell = make_shared<op::RNNCell>(X, W, R, H_t, hidden_size);
    auto function = make_shared<Function>(rnn_cell, ParameterVector{X, W, R, H_t});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    // X
    test_case.add_input<float>(
        {0.3432185f, 0.612268f, 0.20272376f, 0.9513413f, 0.30585995f, 0.7265472f});
    // W
    test_case.add_input<float>({0.41930267f,
                                0.7872176f,
                                0.89940447f,
                                0.23659843f,
                                0.24676207f,
                                0.17101714f,
                                0.3147149f,
                                0.6555601f,
                                0.4559603f});
    // R
    test_case.add_input<float>({0.8374871f,
                                0.86660194f,
                                0.82114047f,
                                0.71549815f,
                                0.18775631f,
                                0.3182116f,
                                0.25392973f,
                                0.38301638f,
                                0.85531586f});
    // Ht
    test_case.add_input<float>(
        {0.12444675f, 0.52055854f, 0.46489045f, 0.4983964f, 0.7730452f, 0.28439692f});

    test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.9408395f, 0.53823817f, 0.84270686f, 0.98932856f, 0.768665f, 0.90461975f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, rnn_cell_bias_clip)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    float clip = 2.88f;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W = make_shared<op::Parameter>(element::f32, Shape{hidden_size, input_size});
    const auto R = make_shared<op::Parameter>(element::f32, Shape{hidden_size, hidden_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{2 * hidden_size});

    const auto rnn_cell = make_shared<op::RNNCell>(X,
                                                   W,
                                                   R,
                                                   H_t,
                                                   hidden_size,
                                                   B,
                                                   vector<string>{"tanh"},
                                                   vector<float>{},
                                                   vector<float>{},
                                                   clip);
    auto function = make_shared<Function>(rnn_cell, ParameterVector{X, W, R, H_t, B});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    // X
    test_case.add_input<float>(
        {0.3432185f, 0.612268f, 0.20272376f, 0.9513413f, 0.30585995f, 0.7265472f});
    // W
    test_case.add_input<float>({0.41930267f,
                                0.7872176f,
                                0.89940447f,
                                0.23659843f,
                                0.24676207f,
                                0.17101714f,
                                0.3147149f,
                                0.6555601f,
                                0.4559603f});
    // R
    test_case.add_input<float>({0.8374871f,
                                0.86660194f,
                                0.82114047f,
                                0.71549815f,
                                0.18775631f,
                                0.3182116f,
                                0.25392973f,
                                0.38301638f,
                                0.85531586f});
    // Ht
    test_case.add_input<float>(
        {0.12444675f, 0.52055854f, 0.46489045f, 0.4983964f, 0.7730452f, 0.28439692f});
    // B
    test_case.add_input<float>(
        {0.45513555f, 0.96227735f, 0.24737759f, 0.57380486f, 0.67398053f, 0.18968852f});

    test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.9922437f, 0.97749525f, 0.9312212f, 0.9937176f, 0.9901317f, 0.95906746f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, rnn_cell_activation_function)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    float clip = 2.88f;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W = make_shared<op::Parameter>(element::f32, Shape{hidden_size, input_size});
    const auto R = make_shared<op::Parameter>(element::f32, Shape{hidden_size, hidden_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{2 * hidden_size});

    const auto rnn_cell = make_shared<op::RNNCell>(X,
                                                   W,
                                                   R,
                                                   H_t,
                                                   hidden_size,
                                                   B,
                                                   vector<string>{"sigmoid"},
                                                   vector<float>{},
                                                   vector<float>{},
                                                   clip);
    auto function = make_shared<Function>(rnn_cell, ParameterVector{X, W, R, H_t, B});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    // X
    test_case.add_input<float>(
        {0.3432185f, 0.612268f, 0.20272376f, 0.9513413f, 0.30585995f, 0.7265472f});
    // W
    test_case.add_input<float>({0.41930267f,
                                0.7872176f,
                                0.89940447f,
                                0.23659843f,
                                0.24676207f,
                                0.17101714f,
                                0.3147149f,
                                0.6555601f,
                                0.4559603f});
    // R
    test_case.add_input<float>({0.8374871f,
                                0.86660194f,
                                0.82114047f,
                                0.71549815f,
                                0.18775631f,
                                0.3182116f,
                                0.25392973f,
                                0.38301638f,
                                0.85531586f});
    // Ht
    test_case.add_input<float>(
        {0.12444675f, 0.52055854f, 0.46489045f, 0.4983964f, 0.7730452f, 0.28439692f});
    // B
    test_case.add_input<float>(
        {0.45513555f, 0.96227735f, 0.24737759f, 0.57380486f, 0.67398053f, 0.18968852f});

    test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.94126844f, 0.9036043f, 0.841243f, 0.9468489f, 0.934215f, 0.873708f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, gru_cell_bias_clip)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;
    float clip = 2.88f;
    bool linear_before_reset = false;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{2 * gates_count * hidden_size});

    const auto gru_cell = make_shared<op::GRUCell>(X,
                                                   W,
                                                   R,
                                                   H_t,
                                                   hidden_size,
                                                   B,
                                                   vector<string>{"sigmoid", "tanh"},
                                                   vector<float>{},
                                                   vector<float>{},
                                                   clip,
                                                   linear_before_reset);
    auto function = make_shared<Function>(gru_cell, ParameterVector{X, W, R, H_t, B});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    // X
    test_case.add_input<float>(
        {0.52421564f, 0.78845507f, 0.9372873f, 0.59783894f, 0.18278378f, 0.2084126f});
    // W
    test_case.add_input<float>(
        {0.5815369f, 0.16559383f, 0.08464007f, 0.843122f,   0.73968244f, 0.11359601f, 0.8295078f,
         0.9240567f, 0.10007995f, 0.20573162f, 0.09002485f, 0.2839569f,  0.3096991f,  0.5638341f,
         0.5787327f, 0.84552664f, 0.16263747f, 0.7243242f,  0.8049057f,  0.43966424f, 0.46294412f,
         0.9833361f, 0.31369713f, 0.1719934f,  0.4937093f,  0.6353004f,  0.77982515f});
    // R
    test_case.add_input<float>(
        {0.16510165f, 0.52435565f, 0.2788478f,  0.99427545f, 0.1623331f,  0.01389796f, 0.99669236f,
         0.53901845f, 0.8737506f,  0.9254788f,  0.21172932f, 0.11634306f, 0.40111724f, 0.37497616f,
         0.2903471f,  0.6796794f,  0.65131867f, 0.78163475f, 0.12058706f, 0.45591718f, 0.791677f,
         0.76497287f, 0.9895242f,  0.7845312f,  0.51267904f, 0.49030215f, 0.08498167f});
    // Ht
    test_case.add_input<float>(
        {0.45738035f, 0.996877f, 0.82882977f, 0.47492632f, 0.88471466f, 0.57833236f});
    // B
    test_case.add_input<float>({0.8286678f,
                                0.9153158f,
                                0.9581612f,
                                0.6639213f,
                                0.84239805f,
                                0.5282445f,
                                0.14153397f,
                                0.22404431f,
                                0.6549655f,
                                0.9175602f,
                                0.14958014f,
                                0.49230585f,
                                0.63162816f,
                                0.4161903f,
                                0.22148274f,
                                0.50496656f,
                                0.34798595f,
                                0.6699164f});

    test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.48588726f, 0.99670005f, 0.83759373f, 0.5023099f, 0.89410484f, 0.60011315f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, gru_cell_linear_before_reset)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;
    float clip = 2.88f;
    bool linear_before_reset = true;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{2 * gates_count * hidden_size});

    const auto gru_cell = make_shared<op::GRUCell>(X,
                                                   W,
                                                   R,
                                                   H_t,
                                                   hidden_size,
                                                   B,
                                                   vector<string>{"sigmoid", "tanh"},
                                                   vector<float>{},
                                                   vector<float>{},
                                                   clip,
                                                   linear_before_reset);
    auto function = make_shared<Function>(gru_cell, ParameterVector{X, W, R, H_t, B});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    // X
    test_case.add_input<float>(
        {0.12249453f, 0.6127907f, 0.5001741f, 0.5124603f, 0.04329684f, 0.023834f});
    // W
    test_case.add_input<float>(
        {0.72259396f, 0.11561195f, 0.9457856f,  0.19037509f, 0.6964006f,  0.33459795f, 0.5468904f,
         0.85646594f, 0.5101311f,  0.9712257f,  0.3687071f,  0.60280246f, 0.56943774f, 0.7475505f,
         0.2490578f,  0.86977345f, 0.85542053f, 0.29660386f, 0.49717373f, 0.7473479f,  0.53454477f,
         0.15974349f, 0.5804805f,  0.14303213f, 0.07514781f, 0.5865731f,  0.76409274f});
    // R
    test_case.add_input<float>(
        {0.91382647f, 0.41527033f, 0.28040004f, 0.23601337f, 0.04471736f, 0.03888785f, 0.06308217f,
         0.44844428f, 0.29384327f, 0.49037653f, 0.50421673f, 0.7366393f,  0.63143945f, 0.00277612f,
         0.37198433f, 0.06966069f, 0.4613444f,  0.10999731f, 0.78273284f, 0.21453214f, 0.10751773f,
         0.18332677f, 0.1326976f,  0.9998985f,  0.19263928f, 0.10979804f, 0.52575564f});
    // Ht
    test_case.add_input<float>(
        {0.8598948f, 0.41189128f, 0.72824323f, 0.53940123f, 0.31485787f, 0.04053852f});
    // B
    test_case.add_input<float>({0.09875853f,
                                0.37801138f,
                                0.7729636f,
                                0.78493553f,
                                0.5662702f,
                                0.12406381f,
                                0.66729516f,
                                0.7752771f,
                                0.78819966f,
                                0.5151927f,
                                0.708666f,
                                0.55303884f,
                                0.03424145f,
                                0.81109315f,
                                0.30524766f,
                                0.6606634f,
                                0.99040645f,
                                0.21112025f});

    test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.8709214f, 0.48411977f, 0.74495184f, 0.6074972f, 0.44572943f, 0.1467715f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, gru_cell_activation_function)
{
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;
    float clip = 2.88f;
    bool linear_before_reset = true;

    const auto X = make_shared<op::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto W =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R =
        make_shared<op::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<op::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{2 * gates_count * hidden_size});

    const auto gru_cell = make_shared<op::GRUCell>(X,
                                                   W,
                                                   R,
                                                   H_t,
                                                   hidden_size,
                                                   B,
                                                   vector<string>{"hardsigmoid", "hardsigmoid"},
                                                   vector<float>{1.8345f, 1.8345f},
                                                   vector<float>{3.05f, 3.05f},
                                                   clip,
                                                   linear_before_reset);
    auto function = make_shared<Function>(gru_cell, ParameterVector{X, W, R, H_t, B});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    // X
    test_case.add_input<float>(
        {0.12249453f, 0.6127907f, 0.5001741f, 0.5124603f, 0.04329684f, 0.023834f});
    // W
    test_case.add_input<float>(
        {0.72259396f, 0.11561195f, 0.9457856f,  0.19037509f, 0.6964006f,  0.33459795f, 0.5468904f,
         0.85646594f, 0.5101311f,  0.9712257f,  0.3687071f,  0.60280246f, 0.56943774f, 0.7475505f,
         0.2490578f,  0.86977345f, 0.85542053f, 0.29660386f, 0.49717373f, 0.7473479f,  0.53454477f,
         0.15974349f, 0.5804805f,  0.14303213f, 0.07514781f, 0.5865731f,  0.76409274f});
    // R
    test_case.add_input<float>(
        {0.91382647f, 0.41527033f, 0.28040004f, 0.23601337f, 0.04471736f, 0.03888785f, 0.06308217f,
         0.44844428f, 0.29384327f, 0.49037653f, 0.50421673f, 0.7366393f,  0.63143945f, 0.00277612f,
         0.37198433f, 0.06966069f, 0.4613444f,  0.10999731f, 0.78273284f, 0.21453214f, 0.10751773f,
         0.18332677f, 0.1326976f,  0.9998985f,  0.19263928f, 0.10979804f, 0.52575564f});
    // Ht
    test_case.add_input<float>(
        {0.8598948f, 0.41189128f, 0.72824323f, 0.53940123f, 0.31485787f, 0.04053852f});
    // B
    test_case.add_input<float>({0.09875853f,
                                0.37801138f,
                                0.7729636f,
                                0.78493553f,
                                0.5662702f,
                                0.12406381f,
                                0.66729516f,
                                0.7752771f,
                                0.78819966f,
                                0.5151927f,
                                0.708666f,
                                0.55303884f,
                                0.03424145f,
                                0.81109315f,
                                0.30524766f,
                                0.6606634f,
                                0.99040645f,
                                0.21112025f});

    test_case.add_expected_output<float>(
        Shape{batch_size, hidden_size},
        {0.8598948f, 0.41189128f, 0.72824323f, 0.53940123f, 0.31485787f, 0.04053852f});

    test_case.run();
}
