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

NGRAPH_TEST(${BACKEND_NAME}, group_conv)
{
    auto data = make_shared<op::v0::Parameter>(element::f32, Shape{1, 4, 2, 2});
    auto filters = make_shared<op::v0::Parameter>(element::f32, Shape{2, 2, 1, 1});
    auto group_conv = make_shared<op::v0::GroupConvolution>(data,
                                                            filters,
                                                            Strides{1, 1},
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1},
                                                            2);
    auto f0 = make_shared<Function>(OutputVector{group_conv}, ParameterVector{data, filters});

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
    auto data = make_shared<op::v0::Parameter>(element::f32, Shape{1, 4, 2, 2});
    auto filters = make_shared<op::v0::Parameter>(element::f32, Shape{2, 2, 1, 1});
    auto group_conv = make_shared<op::v0::GroupConvolution>(data,
                                                            filters,
                                                            Strides{2, 2},
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1},
                                                            2);
    auto f0 = make_shared<Function>(OutputVector{group_conv}, ParameterVector{data, filters});

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
    auto data = make_shared<op::v0::Parameter>(element::f32, Shape{1, 4, 2, 2});
    auto filters = make_shared<op::v0::Parameter>(element::f32, Shape{2, 2, 1, 1});
    auto group_conv = make_shared<op::v0::GroupConvolution>(data,
                                                            filters,
                                                            Strides{1, 1},
                                                            Strides{2, 2},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1},
                                                            2);
    auto f0 = make_shared<Function>(OutputVector{group_conv}, ParameterVector{data, filters});

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
    auto data = make_shared<op::v0::Parameter>(element::f32, Shape{1, 4, 2, 2});
    auto filters = make_shared<op::v0::Parameter>(element::f32, Shape{2, 2, 1, 1});
    auto group_conv = make_shared<op::v0::GroupConvolution>(data,
                                                            filters,
                                                            Strides{1, 1},
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{2, 2},
                                                            2);
    auto f0 = make_shared<Function>(OutputVector{group_conv}, ParameterVector{data, filters});

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
    auto data = make_shared<op::v0::Parameter>(element::f32, Shape{1, 4, 2, 2});
    auto filters = make_shared<op::v0::Parameter>(element::f32, Shape{2, 2, 1, 1});
    auto group_conv = make_shared<op::v0::GroupConvolution>(data,
                                                            filters,
                                                            Strides{1, 1},
                                                            Strides{1, 1},
                                                            CoordinateDiff{1, 0},
                                                            CoordinateDiff{0, 1},
                                                            Strides{1, 1},
                                                            2);
    auto f0 = make_shared<Function>(OutputVector{group_conv}, ParameterVector{data, filters});

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
    auto data = make_shared<op::v0::Parameter>(element::f32, Shape{1, 4, 2, 2});
    auto filters = make_shared<op::v0::Parameter>(element::f32, Shape{2, 2, 1, 1});
    auto group_conv = make_shared<op::v0::GroupConvolution>(data,
                                                            filters,
                                                            Strides{1, 1},
                                                            Strides{2, 2},
                                                            CoordinateDiff{1, 0},
                                                            CoordinateDiff{0, 1},
                                                            Strides{1, 1},
                                                            2);
    auto f0 = make_shared<Function>(OutputVector{group_conv}, ParameterVector{data, filters});

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
    auto data = make_shared<op::v0::Parameter>(element::f32, Shape{1, 4, 4, 1});
    auto filters = make_shared<op::v0::Parameter>(element::f32, Shape{2, 2, 1, 1});
    auto group_conv = make_shared<op::v0::GroupConvolution>(data,
                                                            filters,
                                                            Strides{1, 1},
                                                            Strides{2, 2},
                                                            CoordinateDiff{1, 0},
                                                            CoordinateDiff{0, 1},
                                                            Strides{1, 1},
                                                            2);
    auto f0 = make_shared<Function>(OutputVector{group_conv}, ParameterVector{data, filters});

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
    auto data = make_shared<op::v0::Parameter>(element::f32, Shape{1, 4, 3, 3});
    auto filters = make_shared<op::v0::Parameter>(element::f32, Shape{2, 2, 1, 1});
    auto group_conv = make_shared<op::v0::GroupConvolution>(data,
                                                            filters,
                                                            Strides{1, 1},
                                                            Strides{2, 2},
                                                            CoordinateDiff{1, 0},
                                                            CoordinateDiff{0, 1},
                                                            Strides{1, 1},
                                                            2);
    auto f0 = make_shared<Function>(OutputVector{group_conv}, ParameterVector{data, filters});

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
    auto data = make_shared<op::v0::Parameter>(element::f32, Shape{1, 4, 2, 2});
    auto filters = make_shared<op::v0::Parameter>(element::f32, Shape{2, 1, 2, 1, 1});
    auto group_conv = make_shared<op::v0::GroupConvolution>(data,
                                                            filters,
                                                            Strides{1, 1},
                                                            Strides{1, 1},
                                                            CoordinateDiff{0, 0},
                                                            CoordinateDiff{0, 0},
                                                            Strides{1, 1});
    auto f0 = make_shared<Function>(OutputVector{group_conv}, ParameterVector{data, filters});

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
