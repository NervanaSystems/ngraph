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

NGRAPH_TEST(${BACKEND_NAME}, gather_no_axis)
{
    Shape params_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{3, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::f32, indices_shape);
    auto G = make_shared<op::Gather>(P, I);
    auto f = make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{P}, ParameterVector{I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1.0, 1.2, 2.3, 3.4, 4.5, 5.7});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{0, 1, 1, 2});
    auto result = backend->create_tensor(element::i32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p}, {i});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{2.3, 3.4, 4.5, 5.7}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather)
{
    Shape params_shape{3, 3};
    Shape indices_shape{1, 2};
    Shape out_shape{1, 3, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::f32, indices_shape);
    auto G = make_shared<op::Gather>(P, I, 1);
    auto f = make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{P}, ParameterVector{I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1.0, 1.2, 1.9, 2.3, 3.4, 3.9, 4.5, 5.7, 5.9});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(p, vector<int32_t>{0, 2});
    auto result = backend->create_tensor(element::i32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p}, {i});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{1.0, 1.9, 2.3, 3.9, 4.5, 5.9}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}


NGRAPH_TEST(${BACKEND_NAME}, gather_nd_simple_index_2d)
{
    Shape params_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{3, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::f32, indices_shape);
    auto G = make_shared<op::Gather>(P, I);
    auto f = make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{P}, ParameterVector{I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1.0, 1.2, 2.3, 3.4, 4.5, 5.7});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{0, 1, 1, 2});
    auto result = backend->create_tensor(element::i32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p}, {i});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{2.3, 3.4, 4.5, 5.7}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_batch_index_2d)
{
    Shape params_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{3, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::f32, indices_shape);
    auto G = make_shared<op::Gather>(P, I);
    auto f = make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{P}, ParameterVector{I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1.0, 1.2, 2.3, 3.4, 4.5, 5.7});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{0, 1, 1, 2});
    auto result = backend->create_tensor(element::i32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p}, {i});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{2.3, 3.4, 4.5, 5.7}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_slice_index_2d)
{
    Shape params_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{3, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::f32, indices_shape);
    auto G = make_shared<op::Gather>(P, I);
    auto f = make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{P}, ParameterVector{I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1.0, 1.2, 2.3, 3.4, 4.5, 5.7});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{0, 1, 1, 2});
    auto result = backend->create_tensor(element::i32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p}, {i});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{2.3, 3.4, 4.5, 5.7}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_batch_slice_index_2d)
{
    Shape params_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{3, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::f32, indices_shape);
    auto G = make_shared<op::Gather>(P, I);
    auto f = make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{P}, ParameterVector{I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1.0, 1.2, 2.3, 3.4, 4.5, 5.7});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{0, 1, 1, 2});
    auto result = backend->create_tensor(element::i32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p}, {i});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{2.3, 3.4, 4.5, 5.7}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_index_2d_slice_into_3d)
{
    Shape params_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{3, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::f32, indices_shape);
    auto G = make_shared<op::Gather>(P, I);
    auto f = make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{P}, ParameterVector{I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1.0, 1.2, 2.3, 3.4, 4.5, 5.7});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{0, 1, 1, 2});
    auto result = backend->create_tensor(element::i32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p}, {i});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{2.3, 3.4, 4.5, 5.7}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, gather_nd_batch_3d)
{
    Shape params_shape{3, 2};
    Shape indices_shape{2, 2};
    Shape out_shape{3, 2, 2};
    auto P = make_shared<op::Parameter>(element::f32, params_shape);
    auto I = make_shared<op::Parameter>(element::f32, indices_shape);
    auto G = make_shared<op::Gather>(P, I);
    auto f = make_shared<Function>(make_shared<op::GetOutputElement>(G, 0), ParameterVector{P}, ParameterVector{I});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto p = backend->create_tensor(element::f32, params_shape);
    copy_data(p, vector<float>{1.0, 1.2, 2.3, 3.4, 4.5, 5.7});
    auto i = backend->create_tensor(element::i32, indices_shape);
    copy_data(i, vector<int32_t>{0, 1, 1, 2});
    auto result = backend->create_tensor(element::i32, out_shape);

    auto c = backend->compile(f);
    c->call_with_validate({result}, {p}, {i});
    EXPECT_TRUE(test::all_close_f(
        (vector<float>{2.3, 3.4, 4.5, 5.7}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

