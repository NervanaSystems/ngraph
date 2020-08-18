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
#include <cmath>

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif

#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, mod_3x2_3x2_int32)
{
    Shape shape{3, 2};

    auto A = make_shared<op::v0::Parameter>(element::i32, shape);
    auto B = make_shared<op::v0::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Mod>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{0, 3, 8, 12, 26, 120});
    auto b = backend->create_tensor(element::i32, shape);
    copy_data(b, vector<int32_t>{1, 2, 5, 6, 29, 119});
    auto result = backend->create_tensor(element::i32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ((vector<int32_t>{0, 1, 3, 0, 26, 1}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, mod_3x3_3x3_negative_mix_int32)
{
    Shape shape{3, 3};

    auto A = make_shared<op::v0::Parameter>(element::i32, shape);
    auto B = make_shared<op::v0::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Mod>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{-28, -25, -12, -3, -129, 0, 12, 25, 28});
    auto b = backend->create_tensor(element::i32, shape);
    copy_data(b, vector<int32_t>{7, 11, -13, -3, -120, -2, -13, -12, -14});
    auto result = backend->create_tensor(element::i32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ((vector<int32_t>{0, -3, -12, 0, -9, 0, 12, 1, 0}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, mod_3x2_3x2_f32)
{
    Shape shape{3, 2};

    auto A = make_shared<op::v0::Parameter>(element::f32, shape);
    auto B = make_shared<op::v0::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Mod>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    vector<float> dividend_data{2.f, 0.f, 5.3f, 9.f, 18.5f, 25.3f};
    vector<float> divisor_data{2.f, 2.f, 2.f, 4.5f, 4.2f, 26.8f};
    vector<float> expected_result;
    expected_result.reserve(shape_size(shape));
    for (size_t i = 0; i < shape_size(shape); i++)
    {
        expected_result.push_back(fmod(dividend_data[i], divisor_data[i]));
    }

    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, dividend_data);
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, divisor_data);
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(
        test::all_close_f(expected_result, read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, mod_3x2_scalar_int32)
{
    Shape shape_a{3, 2};
    Shape shape_b{};

    auto A = make_shared<op::v0::Parameter>(element::i32, shape_a);
    auto B = make_shared<op::v0::Parameter>(element::i32, shape_b);
    auto f = make_shared<Function>(make_shared<op::v1::Mod>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{-12, 0, 3, 15, 24, 152});
    auto b = backend->create_tensor(element::i32, shape_b);
    copy_data(b, vector<int32_t>{5});
    auto result = backend->create_tensor(element::i32, shape_a);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ((vector<int32_t>{-2, 0, 3, 0, 4, 2}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, mod_scalar_3x2_int32)
{
    Shape shape_a{};
    Shape shape_b{3, 2};

    auto A = make_shared<op::v0::Parameter>(element::i32, shape_a);
    auto B = make_shared<op::v0::Parameter>(element::i32, shape_b);
    auto f = make_shared<Function>(make_shared<op::v1::Mod>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{25});
    auto b = backend->create_tensor(element::i32, shape_b);
    copy_data(b, vector<int32_t>{-12, 1, 5, 15, 24, 152});
    auto result = backend->create_tensor(element::i32, shape_b);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ((vector<int32_t>{1, 0, 0, 10, 1, 25}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, mod_3_2x2x3_int32)
{
    Shape shape_a{3};
    Shape shape_b{2, 2, 3};

    auto A = make_shared<op::v0::Parameter>(element::i32, shape_a);
    auto B = make_shared<op::v0::Parameter>(element::i32, shape_b);
    auto f = make_shared<Function>(make_shared<op::v1::Mod>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{-6, 29, 120});
    auto b = backend->create_tensor(element::i32, shape_b);
    copy_data(b, vector<int32_t>{-4, -3, 1, 3, 5, 6, 7, 8, 9, 11, 20, 129});
    auto result = backend->create_tensor(element::i32, shape_b);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ((vector<int32_t>{-2, 2, 0, 0, 4, 0, -6, 5, 3, -6, 9, 120}),
              read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, mod_2x2x3_3_int32)
{
    Shape shape_a{2, 2, 3};
    Shape shape_b{3};

    auto A = make_shared<op::v0::Parameter>(element::i32, shape_a);
    auto B = make_shared<op::v0::Parameter>(element::i32, shape_b);
    auto f = make_shared<Function>(make_shared<op::v1::Mod>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{-20, -17, 10, 15, 18, 26, 27, 38, 49, 110, 120, 129});
    auto b = backend->create_tensor(element::i32, shape_b);
    copy_data(b, vector<int32_t>{-6, 8, 9});
    auto result = backend->create_tensor(element::i32, shape_a);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ((vector<int32_t>{-2, -1, 1, 3, 2, 8, 3, 6, 4, 2, 0, 3}),
              read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, mod_2x3x1x1_4x5_int32)
{
    Shape shape_a{2, 3, 1, 1};
    Shape shape_b{4, 5};
    Shape shape_o{2, 3, 4, 5};

    auto A = make_shared<op::v0::Parameter>(element::i32, shape_a);
    auto B = make_shared<op::v0::Parameter>(element::i32, shape_b);
    auto f = make_shared<Function>(make_shared<op::v1::Mod>(A, B), ParameterVector{A, B});

    vector<int32_t> a_data{20, 31, 42, 56, 68, 102};
    vector<int32_t> b_data{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
    vector<int32_t> expected_result{
        0,  2,  0, 0, 2, 6,  4,  2,  0, 9, 8,  7,  6,  5,  4,  3,  2, 1,  0, 20, 1,  1, 3, 1,
        1,  3,  7, 4, 1, 9,  7,  5,  3, 1, 15, 14, 13, 12, 11, 10, 0, 0,  2, 2,  0,  0, 2, 6,
        2,  9,  6, 3, 0, 12, 10, 8,  6, 4, 2,  0,  0,  2,  0,  1,  2, 0,  0, 2,  6,  1, 8, 4,
        0,  11, 8, 5, 2, 18, 16, 14, 0, 2, 0,  3,  2,  5,  4,  5,  8, 2,  8, 3,  12, 8, 4, 0,
        14, 11, 8, 5, 0, 0,  2,  2,  0, 4, 6,  3,  2,  3,  6,  11, 4, 12, 6, 0,  12, 7, 2, 18};

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, a_data);
    auto b = backend->create_tensor(element::i32, shape_b);
    copy_data(b, b_data);
    auto result = backend->create_tensor(element::i32, shape_o);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ(expected_result, read_vector<int32_t>(result));
}
