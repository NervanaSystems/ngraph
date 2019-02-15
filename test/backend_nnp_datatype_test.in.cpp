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
#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>
#include "gtest/gtest.h"

#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/experimental/generate_mask.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/state/rng_state.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/ndarray.hpp"
#include "util/random.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, scalar_constant_int64_bfloat_fail)
{
    auto r = op::Constant::create(element::i64, Shape{}, {501});
    auto f = make_shared<Function>(r, ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::i64, Shape{});

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {});
    EXPECT_EQ(vector<int64_t>{501}, read_vector<int64_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, tensor_constant_int64_bfloat_fail)
{
    Shape shape{2, 2};
    auto r = op::Constant::create(element::i64, shape, {1201, 501, 558, 797});
    auto f = make_shared<Function>(r, ParameterVector{});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto result = backend->create_tensor(element::i64, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {});
    EXPECT_EQ((vector<int64_t>{1201, 501, 558, 797}), read_vector<int64_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, convert_int32_float32_bfloat_fail)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::Convert>(A, element::f32), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{281, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{281, 2, 3, 4}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, divide_int32_bfloat_fail)
{
    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::Divide>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{2, 6000, 8, 16});
    auto b = backend->create_tensor(element::i32, shape);
    copy_data(b, vector<int32_t>{1, 2, 4, 8});
    auto result = backend->create_tensor(element::i32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});
    EXPECT_EQ((vector<int32_t>{2, 3000, 2, 2}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, floor_int32_bfloat_fail)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::Floor>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{-6000, 501, 1, 10});
    auto result = backend->create_tensor(element::i32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<int32_t>{-6000, 501, 1, 10}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, greater_int64_bfloat_fail)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::i64, shape);
    auto B = make_shared<op::Parameter>(element::i64, shape);
    auto f = make_shared<Function>(make_shared<op::Greater>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i64, shape);
    copy_data(a, vector<int64_t>{6000, 1201, -8, 17, -5, 5, 2, 1});
    auto b = backend->create_tensor(element::i64, shape);
    copy_data(b, vector<int64_t>{6000, 1200, 4, 8, 0, 0, 1, 2});
    auto result = backend->create_tensor(element::boolean, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});
    EXPECT_EQ((vector<char>{0, 1, 0, 1, 0, 1, 1, 0}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, lesseq_int32_bfloat_fail)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::LessEq>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{1201, -8, 17, -5});
    auto b = backend->create_tensor(element::i32, shape);
    copy_data(b, vector<int32_t>{1200, 4, 8, 0});
    auto result = backend->create_tensor(element::boolean, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});
    EXPECT_EQ((vector<char>{0, 1, 0, 1}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, maximum_int32_bfloat_fail)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::Maximum>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{1201, 281, -8, 17});
    auto b = backend->create_tensor(element::i32, shape);
    copy_data(b, vector<int32_t>{1200, 280, 4, 8});
    auto result = backend->create_tensor(element::i32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a, b});
    EXPECT_EQ((vector<int32_t>{1201, 281, 4, 17}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_3d_to_scalar_int32_bfloat_fail)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0, 1, 2}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{2,  10, 19, 4,  13, 22, 7,  16, 25, 2,  11, 20, 5, 14,
                                 23, 8,  17, 26, 3,  12, 21, 6,  15, 24, 9,  18, 27});
    auto result = backend->create_tensor(element::i32, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<int32_t>{2 + 10 + 19 + 4 + 13 + 22 + 7 + 16 + 25 + 2 + 11 + 20 + 5 + 14 + 23 +
                               8 + 17 + 26 + 3 + 12 + 21 + 6 + 15 + 24 + 9 + 18 + 27}),
              read_vector<int32_t>(result));
}
