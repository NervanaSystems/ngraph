//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif
// clang-format on

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

NGRAPH_TEST(${BACKEND_NAME}, conv_bias_1d)
{
    auto data = make_shared<op::v0::Parameter>(element::f32, Shape{1, 3, 2});
    auto filters = make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 1});
    auto bias = make_shared<op::v0::Parameter>(element::f32, Shape{2});
    auto conv_bias = make_shared<op::v0::ConvolutionBias>(data, filters, bias);
    auto f0 = make_shared<Function>(OutputVector{conv_bias}, ParameterVector{data, filters, bias});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, Shape{1, 3, 2});
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto b = backend->create_tensor(element::f32, Shape{2, 3, 1});
    copy_data(b, vector<float>{1, 2, 3, 4, 5, 6});
    auto c = backend->create_tensor(element::f32, Shape{2});
    copy_data(c, vector<float>{1, 2});
    auto result0 = backend->create_tensor(element::f32, conv_bias->get_output_shape(0));
    auto handle = backend->compile(f0);
    handle->call_with_validate({result0}, {a, b, c});
    vector<float> expected{23, 29, 51, 66};
    EXPECT_EQ(expected, read_vector<float>(result0));
}

NGRAPH_TEST(${BACKEND_NAME}, conv_bias_2d)
{
    auto data = make_shared<op::v0::Parameter>(element::f32, Shape{1, 3, 2, 2});
    auto filters = make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 1, 1});
    auto bias = make_shared<op::v0::Parameter>(element::f32, Shape{2});
    auto conv_bias = make_shared<op::v0::ConvolutionBias>(data, filters, bias);
    auto f0 = make_shared<Function>(OutputVector{conv_bias}, ParameterVector{data, filters, bias});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, Shape{1, 3, 2, 2});
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto b = backend->create_tensor(element::f32, Shape{2, 3, 1, 1});
    copy_data(b, vector<float>{1, 2, 3, 4, 5, 6});
    auto c = backend->create_tensor(element::f32, Shape{2});
    copy_data(c, vector<float>{1, 2});
    auto result0 = backend->create_tensor(element::f32, conv_bias->get_output_shape(0));
    auto handle = backend->compile(f0);
    handle->call_with_validate({result0}, {a, b, c});
    vector<float> expected{39, 45, 51, 57, 85, 100, 115, 130};
    EXPECT_EQ(expected, read_vector<float>(result0));
}

NGRAPH_TEST(${BACKEND_NAME}, conv_bias_3d)
{
    auto data = make_shared<op::v0::Parameter>(element::f32, Shape{1, 3, 1, 2, 2});
    auto filters = make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 1, 1, 1});
    auto bias = make_shared<op::v0::Parameter>(element::f32, Shape{2});
    auto conv_bias = make_shared<op::v0::ConvolutionBias>(data, filters, bias);
    auto f0 = make_shared<Function>(OutputVector{conv_bias}, ParameterVector{data, filters, bias});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, Shape{1, 3, 1, 2, 2});
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto b = backend->create_tensor(element::f32, Shape{2, 3, 1, 1, 1});
    copy_data(b, vector<float>{1, 2, 3, 4, 5, 6});
    auto c = backend->create_tensor(element::f32, Shape{2});
    copy_data(c, vector<float>{1, 2});
    auto result0 = backend->create_tensor(element::f32, conv_bias->get_output_shape(0));
    auto handle = backend->compile(f0);
    handle->call_with_validate({result0}, {a, b, c});
    vector<float> expected{39, 45, 51, 57, 85, 100, 115, 130};
    EXPECT_EQ(expected, read_vector<float>(result0));
}

NGRAPH_TEST(${BACKEND_NAME}, conv_bias_bprop_2d)
{
    auto data = make_shared<op::v0::Parameter>(element::f32, Shape{1, 3, 2, 2});
    auto filters = make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 1, 1});
    auto bias = make_shared<op::v0::Parameter>(element::f32, Shape{2});
    auto delta = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 2, 2});
    auto conv_bprop =
        make_shared<op::v0::ConvolutionBiasBackpropFiltersBias>(data,
                                                                filters->get_output_shape(0),
                                                                bias->get_output_shape(0),
                                                                delta,
                                                                Strides{1, 1},
                                                                Strides{1, 1},
                                                                CoordinateDiff{0, 0},
                                                                CoordinateDiff{0, 0},
                                                                Strides{1, 1});
    auto goe0 = conv_bprop->output(0);
    auto goe1 = conv_bprop->output(1);
    auto f0 = make_shared<Function>(OutputVector{goe0, goe1}, ParameterVector{data, delta});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, Shape{1, 3, 2, 2});
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto b = backend->create_tensor(element::f32, Shape{1, 2, 2, 2});
    copy_data(b, vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    auto result0 = backend->create_tensor(element::f32, filters->get_output_shape(0));
    auto result1 = backend->create_tensor(element::f32, bias->get_output_shape(0));
    auto handle = backend->compile(f0);
    handle->call_with_validate({result0, result1}, {a, b});
    vector<float> expected0{30, 70, 110, 70, 174, 278};
    vector<float> expected1{10, 26};
    EXPECT_EQ(expected0, read_vector<float>(result0));
    EXPECT_EQ(expected1, read_vector<float>(result1));
}

NGRAPH_TEST(${BACKEND_NAME}, conv_bias_add_2d)
{
    auto data = make_shared<op::v0::Parameter>(element::f32, Shape{1, 3, 2, 2});
    auto filters = make_shared<op::v0::Parameter>(element::f32, Shape{2, 3, 1, 1});
    auto bias = make_shared<op::v0::Parameter>(element::f32, Shape{2});
    auto add = make_shared<op::v0::Parameter>(element::f32, Shape{1, 2, 2, 2});
    auto conv_bias = make_shared<op::v0::ConvolutionBias>(data, filters, bias);
    auto conv_bias_add = make_shared<op::v0::ConvolutionBiasAdd>(conv_bias, add);
    auto f0 = make_shared<Function>(OutputVector{conv_bias_add},
                                    ParameterVector{data, filters, bias, add});

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
    auto result0 = backend->create_tensor(element::f32, conv_bias_add->get_output_shape(0));
    auto handle = backend->compile(f0);
    handle->call_with_validate({result0}, {a, b, c, d});
    vector<float> expected{40, 47, 54, 61, 90, 106, 122, 138};
    EXPECT_EQ(expected, read_vector<float>(result0));
}
