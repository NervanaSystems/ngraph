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
    auto B = make_shared<op::Parameter>(element::f32, Shape{});
    auto elu = make_shared<op::Elu>(A, B);
    auto function = make_shared<Function>(NodeVector{elu}, ParameterVector{A, B});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input(std::vector<float>{-2.f, 3.f, -2.f, 1.f, -1.f, 0.f});
    test_case.add_input(std::vector<float>{0.5f});
    test_case.add_expected_output(
        std::vector<float>{-0.432332358f, 3.f, -0.432332358f, 1.f, -0.316060279f, 0.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, elu_negative_alpha)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{3, 2});
    auto B = make_shared<op::Parameter>(element::f32, Shape{});
    auto elu = make_shared<op::Elu>(A, B);
    auto function = make_shared<Function>(NodeVector{elu}, ParameterVector{A, B});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input(std::vector<float>{-2.f, 3.f, -2.f, 1.f, -1.f, 0.f});
    test_case.add_input(std::vector<float>{-1.f});
    test_case.add_expected_output(
        std::vector<float>{0.864664717f, 3.f, 0.864664717f, 1.f, 0.632120559f, 0.f});
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

NGRAPH_TEST(${BACKEND_NAME}, space_to_depth)
{
    auto A = make_shared<op::Parameter>(element::f32, Shape{1, 2, 4, 4});
    auto space_to_depth = make_shared<op::SpaceToDepth>(A, 2);
    auto function = make_shared<Function>(NodeVector{space_to_depth}, ParameterVector{A});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
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

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
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

NGRAPH_TEST(${BACKEND_NAME}, normalize_across_chw_scalar_scale_4d)
{
    Shape data_shape{1, 2, 3, 4};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto scale = make_shared<op::Parameter>(element::f32, Shape{});
    bool across_spatial{false};
    bool channel_shared{true};
    float eps{1e-6f};

    auto normalize = make_shared<op::Normalize>(data, scale, across_spatial, channel_shared, eps);
    auto function = make_shared<Function>(NodeVector{normalize}, ParameterVector{data, scale});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);
    test_case.add_input<float>({2.f});

    test_case.add_expected_output<float>(
        data_shape, {0.02857143f, 0.05714286f, 0.08571429f, 0.11428571f, 0.14285714f, 0.17142857f,
                     0.2f,        0.22857143f, 0.25714286f, 0.28571429f, 0.31428571f, 0.34285714f,
                     0.37142857f, 0.4f,        0.42857143f, 0.45714286f, 0.48571429f, 0.51428571f,
                     0.54285714f, 0.57142857f, 0.6f,        0.62857143f, 0.65714286f, 0.68571429f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_across_chw_scalar_scale_3d)
{
    Shape data_shape{2, 3, 4};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto scale = make_shared<op::Parameter>(element::f32, Shape{});
    bool across_spatial{false};
    bool channel_shared{true};
    float eps{1e-6f};

    auto normalize = make_shared<op::Normalize>(data, scale, across_spatial, channel_shared, eps);
    auto function = make_shared<Function>(NodeVector{normalize}, ParameterVector{data, scale});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);
    test_case.add_input<float>({2.f});

    test_case.add_expected_output<float>(
        data_shape, {0.02857143f, 0.05714286f, 0.08571429f, 0.11428571f, 0.14285714f, 0.17142857f,
                     0.2f,        0.22857143f, 0.25714286f, 0.28571429f, 0.31428571f, 0.34285714f,
                     0.37142857f, 0.4f,        0.42857143f, 0.45714286f, 0.48571429f, 0.51428571f,
                     0.54285714f, 0.57142857f, 0.6f,        0.62857143f, 0.65714286f, 0.68571429f});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_across_chw_scalar_scale_2d)
{
    Shape data_shape{3, 4};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto scale = make_shared<op::Parameter>(element::f32, Shape{});
    bool across_spatial{false};
    bool channel_shared{true};
    float eps{1e-6f};

    auto normalize = make_shared<op::Normalize>(data, scale, across_spatial, channel_shared, eps);
    auto function = make_shared<Function>(NodeVector{normalize}, ParameterVector{data, scale});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);
    test_case.add_input<float>({2.f});

    test_case.add_expected_output<float>(data_shape,
                                         {0.07844645,
                                          0.15689291,
                                          0.23533936,
                                          0.31378582,
                                          0.39223227,
                                          0.47067872,
                                          0.54912518,
                                          0.62757163,
                                          0.70601809,
                                          0.78446454,
                                          0.86291099,
                                          0.94135745});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, normalize_across_chw_w_scale)
{
    Shape data_shape{1, 2, 3, 4};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto scale = make_shared<op::Parameter>(element::f32, Shape{2});
    bool across_spatial{false};
    bool channel_shared{false};
    float eps{1e-6f};

    auto normalize = make_shared<op::Normalize>(data, scale, across_spatial, channel_shared, eps);
    auto function = make_shared<Function>(NodeVector{normalize}, ParameterVector{data, scale});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);
    test_case.add_input<float>({2.f, 3.f});

    test_case.add_expected_output<float>(
        data_shape, {0.02857143, 0.05714286, 0.08571429, 0.11428571, 0.14285714, 0.17142857,
                     0.2,        0.22857143, 0.25714286, 0.28571429, 0.31428571, 0.34285714,
                     0.55714286, 0.6,        0.64285714, 0.68571429, 0.72857143, 0.77142857,
                     0.81428571, 0.85714286, 0.9,        0.94285714, 0.98571429, 1.02857143});

    test_case.run();
}

// TODO lower tolerance; mismatch at 4th decimal positions
NGRAPH_TEST(DISABLED_${BACKEND_NAME}, normalize_across_hw_w_scale)
{
    Shape data_shape{1, 2, 3, 4};
    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto scale = make_shared<op::Parameter>(element::f32, Shape{2});
    bool across_spatial{true};
    bool channel_shared{false};
    float eps{0.25f};

    auto normalize = make_shared<op::Normalize>(data, scale, across_spatial, channel_shared, eps);
    auto function = make_shared<Function>(NodeVector{normalize}, ParameterVector{data, scale});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");

    vector<float> input_data(shape_size(data_shape));
    iota(begin(input_data), end(input_data), 1);

    test_case.add_input<float>(input_data);
    test_case.add_input<float>({2.f, 3.f});

    test_case.add_expected_output<float>(
        data_shape, {0.07844646, 0.15689291, 0.23533936, 0.31378582, 0.39223227, 0.47067872,
                     0.5491252,  0.62757164, 0.7060181,  0.78446454, 0.862911,   0.94135743,
                     0.5982327,  0.64425063, 0.6902685,  0.7362864,  0.7823043,  0.8283222,
                     0.87434006, 0.920358,   0.9663758,  1.0123938,  1.0584116,  1.1044296});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, gemm)
{
    auto A = make_shared<op::Parameter>(element::f64, Shape{3, 6});
    auto B = make_shared<op::Parameter>(element::f64, Shape{6, 4});
    auto C = make_shared<op::Parameter>(element::f64, Shape{3, 4});

    auto gemm_func = make_shared<op::Gemm>(A, B, C);
    auto function = make_shared<Function>(NodeVector{gemm_func}, ParameterVector{A, B, C});
    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    // A
    test_case.add_input<double>(vector<double>(18, 1));
    // B
    test_case.add_input<double>(vector<double>(24, 2));
    // C
    test_case.add_input<double>(vector<double>(12, 0));
    //output
    test_case.add_expected_output<double>(Shape{3, 4}, vector<double>(12, 12));
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, gemm_broadcast_input_C)
{
    auto A = make_shared<op::Parameter>(element::f64, Shape{3, 6});
    auto B = make_shared<op::Parameter>(element::f64, Shape{6, 4});
    auto C = make_shared<op::Parameter>(element::f64, Shape{});

    auto gemm_func = make_shared<op::Gemm>(A, B, C, 0.5);
    auto function = make_shared<Function>(NodeVector{gemm_func}, ParameterVector{A, B, C});
    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    // A
    test_case.add_input<double>(vector<double>(18, 1));
    // B
    test_case.add_input<double>(vector<double>(24, 2));
    // C
    test_case.add_input<double>(vector<double>{1});
    //output
    test_case.add_expected_output<double>(Shape{3, 4}, vector<double>(12, 7));
}

NGRAPH_TEST(${BACKEND_NAME}, fused_clamp)
{
    auto data = make_shared<op::Parameter>(element::f64, Shape{4, 4});
    auto tested_op = make_shared<op::Clamp>(data, 10.0, 20.0);
    auto function = make_shared<Function>(tested_op, ParameterVector{data});

    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    test_case.add_input<double>({std::numeric_limits<double>::min(),
                                 std::numeric_limits<double>::max(),
                                 -std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::infinity(),
                                 -1.0,
                                 0.0,
                                 1.0,
                                 9.99999,
                                 10.0,
                                 10.0000001,
                                 15.0,
                                 19.9999999,
                                 20.0,
                                 20.0000001,
                                 21.0,
                                 100.0});

    test_case.add_expected_output<double>(Shape{4, 4},
                                          {10.0,
                                           20.0,
                                           10.0,
                                           20.0,
                                           10.0,
                                           10.0,
                                           10.0,
                                           10.0,
                                           10.0,
                                           10.0000001,
                                           15.0,
                                           19.9999999,
                                           20.0,
                                           20.0,
                                           20.0,
                                           20.0});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, mvn_mean_normalization)
{
    Shape data_shape{1, 2, 5};
    auto data = make_shared<op::Parameter>(element::f64, data_shape);

    auto mvn_func = make_shared<op::MVN>(data, true, false);
    auto function = make_shared<Function>(NodeVector{mvn_func}, ParameterVector{data});
    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    // data
    vector<double> data_vector(shape_size(data_shape));
    iota(begin(data_vector), end(data_vector), 0);
    test_case.add_input<double>(data_vector);

    // expected result
    test_case.add_expected_output<double>(
        data_shape, vector<double>{-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, mvn_mean_normalization_split_channels)
{
    Shape data_shape{1, 2, 5, 1};
    auto data = make_shared<op::Parameter>(element::f64, data_shape);

    auto mvn_func = make_shared<op::MVN>(data, false, false);
    auto function = make_shared<Function>(NodeVector{mvn_func}, ParameterVector{data});
    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    // data
    vector<double> data_vector(shape_size(data_shape));
    iota(begin(data_vector), end(data_vector), 0);
    test_case.add_input<double>(data_vector);

    // expected result
    test_case.add_expected_output<double>({1, 2, 5, 1},
                                          vector<double>{-2, -1, 0, 1, 2, -2, -1, 0, 1, 2});

    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, mvn_mean_variance_normalization)
{
    Shape data_shape{1, 2, 5};
    auto data = make_shared<op::Parameter>(element::f64, data_shape);

    auto mvn_func = make_shared<op::MVN>(data);
    auto function = make_shared<Function>(NodeVector{mvn_func}, ParameterVector{data});
    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    // data
    vector<double> data_vector(shape_size(data_shape));
    iota(begin(data_vector), end(data_vector), 0);
    test_case.add_input<double>(data_vector);

    // expected result
    test_case.add_expected_output<double>(data_shape,
                                          vector<double>{-1.566698903055826,
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
    auto data = make_shared<op::Parameter>(element::f64, data_shape);

    auto mvn_func = make_shared<op::MVN>(data, false);
    auto function = make_shared<Function>(NodeVector{mvn_func}, ParameterVector{data});
    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    // data
    vector<double> data_vector(shape_size(data_shape));
    iota(begin(data_vector), end(data_vector), 0);
    test_case.add_input<double>(data_vector);

    // expected result
    test_case.add_expected_output<double>(data_shape,
                                          vector<double>{-1.4142135613730948,
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

NGRAPH_TEST(${BACKEND_NAME}, scale_shift_no_broadcast)
{
    auto data = make_shared<op::Parameter>(element::f64, Shape{3, 6});
    auto scale = make_shared<op::Parameter>(element::f64, Shape{3, 6});
    auto shift = make_shared<op::Parameter>(element::f64, Shape{3, 6});

    auto scale_shift_func = make_shared<op::ScaleShift>(data, scale, shift);
    auto function =
        make_shared<Function>(NodeVector{scale_shift_func}, ParameterVector{data, scale, shift});
    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    // Data
    test_case.add_input<double>(vector<double>(18, 2));
    // Scale
    test_case.add_input<double>(vector<double>(18, 2));
    // Shift
    test_case.add_input<double>(vector<double>(18, 2));
    //output
    test_case.add_expected_output<double>(Shape{3, 6}, vector<double>(18, 6));
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, scale_shift)
{
    auto data = make_shared<op::Parameter>(element::f64, Shape{3, 6});
    auto scale = make_shared<op::Parameter>(element::f64, Shape{3, 6});
    auto shift = make_shared<op::Parameter>(element::f64, Shape{});

    auto scale_shift_func = make_shared<op::ScaleShift>(data, scale, shift);
    auto function =
        make_shared<Function>(NodeVector{scale_shift_func}, ParameterVector{data, scale, shift});
    auto test_case = ngraph::test::NgraphTestCase(function, "${BACKEND_NAME}");
    // Data
    test_case.add_input<double>(vector<double>(18, 2));
    // Scale
    test_case.add_input<double>(vector<double>(18, 2));
    // Shift
    test_case.add_input<double>(vector<double>{2});
    //output
    test_case.add_expected_output<double>(Shape{3, 6}, vector<double>(18, 6));
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
