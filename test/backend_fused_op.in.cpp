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
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

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

NGRAPH_TEST(${BACKEND_NAME}, hardsigmoid_float)
{
    Shape shape{3, 5};
    float alpha = 0.125f;
    float beta = 0.642f;

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto hardsigmoid = make_shared<op::HardSigmoid>(A, alpha, beta);
    auto f0 = make_shared<Function>(NodeVector{hardsigmoid}, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Prepare input and expected output data
    std::vector<float> input_data{-1.f,
                                  0.f,
                                  1.f,
                                  -100.f,
                                  100.f,
                                  -3.1234567f,
                                  5.876543f,
                                  7.13245364f,
                                  std::numeric_limits<float>::max(),
                                  std::numeric_limits<float>::lowest(),
                                  std::numeric_limits<float>::min(),
                                  -std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::min() / 16.f,
                                  -std::numeric_limits<float>::min() / 16.f};

    auto impl = [alpha, beta](float val) {
        return std::min(std::max(alpha * val + beta, 0.f), 1.f);
    };
    std::vector<float> expected_output;
    std::transform(
        std::begin(input_data), std::end(input_data), std::back_inserter(expected_output), impl);

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, input_data);
    auto result0 = backend->create_tensor(element::f32, shape);
    auto handle = backend->compile(f0);
    handle->call_with_validate({result0}, {a});

    EXPECT_TRUE(test::all_close_f(expected_output, read_vector<float>(result0)));
}

NGRAPH_TEST(${BACKEND_NAME}, hardsigmoid_double)
{
    Shape shape{3, 5};
    double alpha = 0.125;
    double beta = 0.642;

    auto A = make_shared<op::Parameter>(element::f64, shape);
    auto hardsigmoid = make_shared<op::HardSigmoid>(A, alpha, beta);
    auto f0 = make_shared<Function>(NodeVector{hardsigmoid}, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Prepare input and expected output data
    std::vector<double> input_data{-1.,
                                   0.,
                                   1.,
                                   -100.,
                                   100.,
                                   -3.1234567,
                                   5.876543,
                                   7.13245364,
                                   std::numeric_limits<double>::max(),
                                   std::numeric_limits<double>::lowest(),
                                   std::numeric_limits<double>::min(),
                                   -std::numeric_limits<double>::infinity(),
                                   std::numeric_limits<double>::infinity(),
                                   std::numeric_limits<double>::min() / 16.,
                                   -std::numeric_limits<double>::min() / 16.};

    auto impl = [alpha, beta](double val) {
        return std::min(std::max(alpha * val + beta, 0.), 1.);
    };
    std::vector<double> expected_output;
    std::transform(
        std::begin(input_data), std::end(input_data), std::back_inserter(expected_output), impl);

    auto a = backend->create_tensor(element::f64, shape);
    copy_data(a, input_data);
    auto result0 = backend->create_tensor(element::f64, shape);
    auto handle = backend->compile(f0);
    handle->call_with_validate({result0}, {a});

    EXPECT_TRUE(test::all_close_f(expected_output, read_vector<double>(result0)));
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
