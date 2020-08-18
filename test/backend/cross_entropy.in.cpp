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

NGRAPH_TEST(${BACKEND_NAME}, cross_entropy_with_soft_labels)
{
    Shape tensor_shape{2, 4};
    auto input = make_shared<op::v0::Parameter>(element::f32, tensor_shape);
    auto labels = make_shared<op::v0::Parameter>(element::i32, Shape{2, 4});
    auto cross_entropy = make_shared<op::v0::CrossEntropy>(input, labels, true);
    auto f0 = make_shared<Function>(OutputVector{cross_entropy}, ParameterVector{input, labels});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, tensor_shape);
    copy_data(a, vector<float>{0.25f, 0.25f, 0.25f, 0.25f, 0.01f, 0.01f, 0.01f, 0.96f});
    auto b = backend->create_tensor(element::i32, Shape{2, 4});
    copy_data(b, vector<int32_t>{0, 0, 0, 1, 0, 0, 0, 1});
    auto result0 = backend->create_tensor(element::f32, Shape{2, 1});
    auto handle = backend->compile(f0);
    handle->call_with_validate({result0}, {a, b});
    vector<float> expected{1.38629f, 0.040822f};
    auto result = read_vector<float>(result0);
    EXPECT_TRUE(test::all_close_f(result, expected, 23));
}

NGRAPH_TEST(${BACKEND_NAME}, cross_entropy_with_one_hot)
{
    Shape tensor_shape{2, 4};
    auto input = make_shared<op::v0::Parameter>(element::f32, tensor_shape);
    auto labels = make_shared<op::v0::Parameter>(element::i32, Shape{2, 1});
    auto cross_entropy = make_shared<op::v0::CrossEntropy>(input, labels, false);
    auto f0 = make_shared<Function>(OutputVector{cross_entropy}, ParameterVector{input, labels});
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, tensor_shape);
    copy_data(a, vector<float>{0.25f, 0.25f, 0.25f, 0.25f, 0.01f, 0.01f, 0.01f, 0.96f});
    auto b = backend->create_tensor(element::i32, Shape{2, 1});
    copy_data(b, vector<int32_t>{1, 1});
    auto result0 = backend->create_tensor(element::f32, Shape{2, 1});
    auto handle = backend->compile(f0);
    handle->call_with_validate({result0}, {a, b});
    vector<float> expected{1.38629f, 4.60517f};
    auto result = read_vector<float>(result0);
    EXPECT_TRUE(test::all_close_f(result, expected, 23));
}
