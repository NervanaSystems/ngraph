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

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/known_element_types.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, relu_2Dfprop)
{
    auto shape_a = Shape{2, 5};
    auto A = make_shared<op::v0::Parameter>(element::f32, shape_a);
    auto relu = make_shared<op::v0::Relu>(A);
    auto shape_rt = Shape{2, 5};
    auto f = make_shared<Function>(relu, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5});
    auto result = backend->create_tensor(element::f32, shape_rt);
    vector<float> expected{1, 8, 0, 17, 0, 1, 8, 0, 17, 0};

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(read_vector<float>(result), expected, MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, relu_2Dfprop_i32)
{
    auto shape_a = Shape{2, 5};
    auto A = make_shared<op::v0::Parameter>(element::i32, shape_a);
    auto relu = make_shared<op::v0::Relu>(A);
    auto shape_rt = Shape{2, 5};
    auto f = make_shared<Function>(relu, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{1, 8, -8, 17, -2, 1, 8, -8, 17, -1});
    auto result = backend->create_tensor(element::i32, shape_rt);
    vector<int32_t> expected{1, 8, 0, 17, 0, 1, 8, 0, 17, 0};

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ(expected, read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, relu_4Dfprop)
{
    auto shape_a = Shape{2, 2, 2, 2};
    auto A = make_shared<op::v0::Parameter>(element::f32, shape_a);
    auto relu = make_shared<op::v0::Relu>(A);
    auto shape_rt = Shape{2, 2, 2, 2};
    auto f = make_shared<Function>(relu, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1});
    auto result = backend->create_tensor(element::f32, shape_rt);
    vector<float> expected{1, 8, 0, 17, 0, 1, 8, 0, 17, 0, 1, 8, 0, 17, 0, 1};

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(read_vector<float>(result), expected, MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, fuse_max_with_constant_zero_input_as_relu)
{
    auto shape_a = Shape{2, 5};
    auto A = op::v0::Constant::create(element::f32, shape_a, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    auto B = make_shared<op::v0::Parameter>(element::f32, shape_a);
    auto max = make_shared<op::v1::Maximum>(A, B);
    auto shape_rt = Shape{2, 5};
    auto f = make_shared<Function>(max, ParameterVector{B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto b = backend->create_tensor(element::f32, shape_a);
    copy_data(b, vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5});
    auto result = backend->create_tensor(element::f32, shape_rt);
    vector<float> expected{1, 8, 0, 17, 0, 1, 8, 0, 17, 0};

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {b});
    EXPECT_TRUE(test::all_close_f(read_vector<float>(result), expected, MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, relu_2Dbackprop)
{
    auto shape_a = Shape{2, 5};
    auto A = make_shared<op::v0::Parameter>(element::f32, shape_a);
    auto delta_val = make_shared<op::v0::Parameter>(element::f32, shape_a);
    auto relu = make_shared<op::v0::ReluBackprop>(A, delta_val);
    auto shape_rt = Shape{2, 5};
    auto f = make_shared<Function>(relu, ParameterVector{A, delta_val});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5});
    auto delta = backend->create_tensor(element::f32, shape_a);
    copy_data(delta, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
    auto result = backend->create_tensor(element::f32, shape_rt);
    vector<float> expected{1, 2, 0, 4, 0, 6, 7, 0, 9, 0};

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, delta});
    EXPECT_TRUE(test::all_close_f(read_vector<float>(result), expected, MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, relu_4Dbackprop)
{
    auto shape_a = Shape{2, 2, 2, 2};
    auto A = make_shared<op::v0::Parameter>(element::f32, shape_a);
    auto delta_val = make_shared<op::v0::Parameter>(element::f32, shape_a);
    auto relu = make_shared<op::v0::ReluBackprop>(A, delta_val);
    auto shape_rt = Shape{2, 2, 2, 2};
    auto f = make_shared<Function>(relu, ParameterVector{A, delta_val});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1});
    auto delta = backend->create_tensor(element::f32, shape_a);
    copy_data(delta, vector<float>{1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1, 8, -8, 17, -0.5, 1});
    auto result = backend->create_tensor(element::f32, shape_rt);
    vector<float> expected{1, 8, 0, 17, 0, 1, 8, 0, 17, 0, 1, 8, 0, 17, 0, 1};

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, delta});
    EXPECT_TRUE(test::all_close_f(read_vector<float>(result), expected, MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, relu_unsigned_limit)
{
    auto shape = Shape{};
    auto A = make_shared<op::v0::Parameter>(element::u32, shape);
    auto relu = make_shared<op::v0::Relu>(A);
    auto f = make_shared<Function>(relu, ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::u32, shape);
    copy_data(a, vector<uint32_t>{std::numeric_limits<uint32_t>::max()});
    auto result = backend->create_tensor(element::u32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_EQ((vector<uint32_t>{std::numeric_limits<uint32_t>::max()}),
              read_vector<uint32_t>(result));
}
