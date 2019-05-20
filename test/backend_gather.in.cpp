//*****************************************************************************
// Copyright 2019 Intel Corporation
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
#include <random>
#include <string>

#include "gtest/gtest.h"
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

NGRAPH_TEST(${BACKEND_NAME}, gather_no_axis)
{
    Shape params_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::Gather>(P, I);
    auto f = make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1.0, 1.1, 2.0, 2.1, 3.0, 3.1});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{0, 1, 1, 2});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f((vector<float>{1.0, 1.1, 2.0, 2.1, 2.0, 2.1, 3.0, 3.1}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_scalar_indices_no_axis)
{
    Shape params_shape{3, 2};
    Shape indices_shape{};
    Shape out_shape{2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::Gather>(P, I);
    auto f = make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1.0, 1.1, 2.0, 2.1, 3.0, 3.1});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{1});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{2.0, 2.1}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather)
{
    Shape params_shape{3, 3};
    Shape indices_shape{1, 2};
    Shape out_shape{3, 1, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::Gather>(P, I, 1);
    auto f = make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{0, 2});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f((vector<float>{1.0, 1.2, 2.0, 2.2, 3.0, 3.2}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_scalar_indices)
{
    Shape params_shape{3, 3};
    Shape indices_shape{};
    Shape out_shape{3};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::Gather>(P, I, 1);
    auto f = make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{0});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1.0, 2.0, 3.0}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_single_indices)
{
    Shape params_shape{3, 3};
    Shape indices_shape{2};
    Shape out_shape{};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(P, I);
    auto f = make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{1, 2});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1.5}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_scalar_from_2d)
{
    Shape params_shape{2, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(P, I);
    auto f = make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1.0, 1.1, 1.2, 1.3});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{0, 0, 1, 1});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1.0, 1.3}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_1d_from_2d)
{
    Shape params_shape{2, 2};
    Shape indices_shape{2, 1};
    Shape out_shape{2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(P, I);
    auto f = make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1.0, 1.1, 1.2, 1.3});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{1, 0});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1.2, 1.3, 1.0, 1.1}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_scalar_from_3d)
{
    Shape params_shape{2, 2, 2};
    Shape indices_shape{2, 3};
    Shape out_shape{2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(P, I);
    auto f = make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{0, 0, 1, 1, 0, 1});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1.1, 2.1}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_1d_from_3d)
{
    Shape params_shape{2, 2, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(P, I);
    auto f = make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{0, 1, 1, 0});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1.2, 1.3, 2.0, 2.1}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_2d_from_3d)
{
    Shape params_shape{2, 2, 2};
    Shape indices_shape{1, 1};
    Shape out_shape{1, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(P, I);
    auto f = make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{1});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{2.0, 2.1, 2.2, 2.3}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_batch_scalar_from_2d)
{
    Shape params_shape{2, 2};
    Shape indices_shape{2, 1, 2};
    Shape out_shape{2, 1};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(P, I);
    auto f = make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1.0, 1.1, 1.2, 1.3});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{0, 0, 0, 1});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1.0, 1.1}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_batch_1d_from_2d)
{
    Shape params_shape{2, 2};
    Shape indices_shape{2, 1, 1};
    Shape out_shape{2, 1, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(P, I);
    auto f = make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1.0, 1.1, 1.2, 1.3});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{1, 0});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1.2, 1.3, 1.0, 1.1}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_batch_scalar_from_3d)
{
    Shape params_shape{2, 2, 2};
    Shape indices_shape{2, 2, 3};
    Shape out_shape{2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(P, I);
    auto f = make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1.1, 2.1, 1.3, 2.2}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_batch_1d_from_3d)
{
    Shape params_shape{2, 2, 2};
    Shape indices_shape{2, 2, 2};
    Shape out_shape{2, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(P, I);
    auto f = make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{0, 1, 1, 0, 0, 0, 1, 1});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f((vector<float>{1.2, 1.3, 2.0, 2.1, 1.0, 1.1, 2.2, 2.3}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_batch_2d_from_3d)
{
    Shape params_shape{2, 2, 2};
    Shape indices_shape{2, 1, 1};
    Shape out_shape{2, 1, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::GatherND>(P, I);
    auto f = make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{1, 0});
    auto result = backend->create_tensor(element::f32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close_f((vector<float>{2.0, 2.1, 2.2, 2.3, 1.0, 1.1, 1.2, 1.3}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_no_axis_int8)
{
    Shape params_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto P = make_shared<op::Parameter>(element::i8, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::Gather>(P, I);
    auto f = make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::i8, params_shape);
    copy_data(p, vector<int8_t>{10, 11, 20, 21, 30, 31});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{0, 1, 1, 2});
    auto result = backend->create_tensor(element::i8, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close((vector<int8_t>{10, 11, 20, 21, 20, 21, 30, 31}),
                                read_vector<int8_t>(result),
                                static_cast<int8_t> MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_no_axis_int16)
{
    Shape params_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto P = make_shared<op::Parameter>(element::i16, params_shape);
    auto I = make_shared<op::Parameter>(element::i64, indices_shape);
    auto G = make_shared<op::Gather>(P, I);
    auto f = make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::i16, params_shape);
    copy_data(p, vector<int16_t>{10, 11, 20, 21, 30, 31});
    auto i = backend->create_tensor(element::i64, indices_shape);
    copy_data(i, vector<int64_t>{0, 1, 1, 2});
    auto result = backend->create_tensor(element::i16, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close((vector<int16_t>{10, 11, 20, 21, 20, 21, 30, 31}),
                                read_vector<int16_t>(result),
                                static_cast<int16_t> MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_no_axis_int32)
{
    Shape params_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto P = make_shared<op::Parameter>(element::i32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::Gather>(P, I);
    auto f = make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::i32, params_shape);
    copy_data(p, vector<int32_t>{10, 11, 20, 21, 30, 31});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{0, 1, 1, 2});
    auto result = backend->create_tensor(element::i32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close((vector<int32_t>{10, 11, 20, 21, 20, 21, 30, 31}),
                                read_vector<int32_t>(result),
                                MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_no_axis_int64)
{
    Shape params_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto P = make_shared<op::Parameter>(element::i64, params_shape);
    auto I = make_shared<op::Parameter>(element::i64, indices_shape);
    auto G = make_shared<op::Gather>(P, I);
    auto f = make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::i64, params_shape);
    copy_data(p, vector<int64_t>{10, 11, 20, 21, 30, 31});
    auto i = backend->create_tensor(element::i64, indices_shape);
    copy_data(i, vector<int64_t>{0, 1, 1, 2});
    auto result = backend->create_tensor(element::i64, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close((vector<int64_t>{10, 11, 20, 21, 20, 21, 30, 31}),
                                read_vector<int64_t>(result),
                                static_cast<int64_t> MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_no_axis_uint8)
{
    Shape params_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto P = make_shared<op::Parameter>(element::u8, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::Gather>(P, I);
    auto f = make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::u8, params_shape);
    copy_data(p, vector<uint8_t>{10, 11, 20, 21, 30, 31});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{0, 1, 1, 2});
    auto result = backend->create_tensor(element::u8, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close((vector<uint8_t>{10, 11, 20, 21, 20, 21, 30, 31}),
                                read_vector<uint8_t>(result),
                                static_cast<uint8_t> MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_no_axis_uint16)
{
    Shape params_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto P = make_shared<op::Parameter>(element::u16, params_shape);
    auto I = make_shared<op::Parameter>(element::i64, indices_shape);
    auto G = make_shared<op::Gather>(P, I);
    auto f = make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::u16, params_shape);
    copy_data(p, vector<uint16_t>{10, 11, 20, 21, 30, 31});
    auto i = backend->create_tensor(element::i64, indices_shape);
    copy_data(i, vector<int64_t>{0, 1, 1, 2});
    auto result = backend->create_tensor(element::u16, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close((vector<uint16_t>{10, 11, 20, 21, 20, 21, 30, 31}),
                                read_vector<uint16_t>(result),
                                static_cast<uint16_t> MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_no_axis_uint32)
{
    Shape params_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto P = make_shared<op::Parameter>(element::u32, params_shape);
    auto I = make_shared<op::Parameter>(element::i32, indices_shape);
    auto G = make_shared<op::Gather>(P, I);
    auto f = make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::u32, params_shape);
    copy_data(p, vector<uint32_t>{10, 11, 20, 21, 30, 31});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{0, 1, 1, 2});
    auto result = backend->create_tensor(element::u32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close((vector<uint32_t>{10, 11, 20, 21, 20, 21, 30, 31}),
                                read_vector<uint32_t>(result),
                                static_cast<uint32_t> MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_no_axis_uint64)
{
    Shape params_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto P = make_shared<op::Parameter>(element::u64, params_shape);
    auto I = make_shared<op::Parameter>(element::i64, indices_shape);
    auto G = make_shared<op::Gather>(P, I);
    auto f = make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::u64, params_shape);
    copy_data(p, vector<uint64_t>{10, 11, 20, 21, 30, 31});
    auto i = backend->create_tensor(element::i64, indices_shape);
    copy_data(i, vector<int64_t>{0, 1, 1, 2});
    auto result = backend->create_tensor(element::u64, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close((vector<uint64_t>{10, 11, 20, 21, 20, 21, 30, 31}),
                                read_vector<uint64_t>(result),
                                static_cast<uint64_t> MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_no_axis_bool)
{
    Shape params_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{2, 2, 2};
    auto P = make_shared<op::Parameter>(element::boolean, params_shape);
    auto I = make_shared<op::Parameter>(element::i64, indices_shape);
    auto G = make_shared<op::Gather>(P, I);
    auto f = make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{P, I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::boolean, params_shape);
    copy_data(p, vector<char>{1, 1, 1, 0, 0, 1});
    auto i = backend->create_tensor(element::i64, indices_shape);
    copy_data(i, vector<int64_t>{0, 1, 1, 2});
    auto result = backend->create_tensor(element::boolean, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p, i});
    EXPECT_TRUE(test::all_close((vector<char>{1, 1, 1, 0, 1, 0, 0, 1}),
                                read_vector<char>(result),
                                static_cast<char> MIN_FLOAT_TOLERANCE_BITS));
}
