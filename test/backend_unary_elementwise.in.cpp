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
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, abs)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Abs>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, -2, 0, -4.75f});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 0, 4.75f}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, acos)
{
    Shape shape{11};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Acos>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> input{-1.f, -0.75f, -0.5f, -0.25f, -0.125f, 0.f, 0.125f, 0.25f, 0.5f, 0.75f, 1.f};
    copy_data(a, input);
    auto result = backend->create_tensor(element::f32, shape);
    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_TRUE(test::all_close_f(vector<float>{3.14159265f,
                                                2.41885841f,
                                                2.09439510f,
                                                1.82347658f,
                                                1.69612416f,
                                                1.57079633f,
                                                1.44546850f,
                                                1.31811607f,
                                                1.04719755f,
                                                0.72273425f,
                                                0.00000000f},
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, asin)
{
    Shape shape{11};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Asin>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> input{-1.f, -0.75f, -0.5f, -0.25f, -0.125f, 0.f, 0.125f, 0.25f, 0.5f, 0.75f, 1.f};
    copy_data(a, input);
    auto result = backend->create_tensor(element::f32, shape);
    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_TRUE(test::all_close_f(vector<float>{-1.57079633f,
                                                -0.84806208f,
                                                -0.52359878f,
                                                -0.25268026f,
                                                -0.12532783f,
                                                0.00000000f,
                                                0.12532783f,
                                                0.25268026f,
                                                0.52359878f,
                                                0.84806208f,
                                                1.57079633f},
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, atan)
{
    Shape shape{11};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Atan>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> input{-4.f, -2.f, -1.f, -0.5f, -0.25f, 0.f, 0.25f, 0.5f, 1.f, 2.f, 4.f};
    copy_data(a, input);
    auto result = backend->create_tensor(element::f32, shape);
    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_TRUE(test::all_close_f(vector<float>{-1.32581766f,
                                                -1.10714872f,
                                                -0.78539816f,
                                                -0.46364761f,
                                                -0.24497866f,
                                                0.00000000f,
                                                0.24497866f,
                                                0.46364761f,
                                                0.78539816f,
                                                1.10714872f,
                                                1.32581766f},
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, ceiling)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Ceiling>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{-2.5f, -2.0f, 0.3f, 4.8f});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{-2.0f, -2.0f, 1.0f, 5.0f}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, cos)
{
    Shape shape{11};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Cos>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> input{0.f, 0.25f, -0.25f, 0.5f, -0.5f, 1.f, -1.f, 2.f, -2.f, 4.f, -4.f};
    copy_data(a, input);
    auto result = backend->create_tensor(element::f32, shape);
    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_TRUE(test::all_close_f(vector<float>{1.00000000f,
                                                0.96891242f,
                                                0.96891242f,
                                                0.87758256f,
                                                0.87758256f,
                                                0.54030231f,
                                                0.54030231f,
                                                -0.41614684f,
                                                -0.41614684f,
                                                -0.65364362f,
                                                -0.65364362f},
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, cosh)
{
    Shape shape{6};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Cosh>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 5.0f, -5.0f};
    copy_data(a, input);
    auto result = backend->create_tensor(element::f32, shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return coshf(x); });

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_TRUE(test::all_close_f(input, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, exp)
{
    Shape shape{8};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Exp>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{-4, -3, -2, -1, 0, 1, 2, 3});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_TRUE(test::all_close_f(
        vector<float>{expf(-4), expf(-3), expf(-2), expf(-1), expf(0), expf(1), expf(2), expf(3)},
        read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, floor)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Floor>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{-2.5f, -2.0f, 0.3f, 4.8f});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{-3.0f, -2.0f, 0.0f, 4.0f}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, floor_int32)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::Floor>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{-2, -136314880, 0, 4});
    auto result = backend->create_tensor(element::i32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<int32_t>{-2, -136314880, 0, 4}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, log)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Log>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{0.125f, 0.25f, 0.5f, 1.f, 2.f, 4.f, 8.f, 16.f});
    vector<float> loga{-2.07944154f,
                       -1.38629436f,
                       -0.69314718f,
                       0.00000000f,
                       0.69314718f,
                       1.38629436f,
                       2.07944154f,
                       2.77258872f};
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_TRUE(test::all_close_f(loga, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, negative)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Negative>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, -2, 0, -4.75f, 8.75f, -8.75f});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{-1, 2, 0, 4.75f, -8.75f, 8.75f}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, not)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::boolean, shape);
    auto f = make_shared<Function>(make_shared<op::Not>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::boolean, shape);
    copy_data(a, vector<char>{1, 0, 2, 0});
    auto result = backend->create_tensor(element::boolean, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<char>{0, 1, 0, 1}), read_vector<char>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, sign)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sign>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, -2, 0, -4.8f, 4.8f, -0.0f});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{1, -1, 0, -1, 1, 0}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, sin)
{
    Shape shape{11};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sin>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> input{0.f, 0.25f, -0.25f, 0.5f, -0.5f, 1.f, -1.f, 2.f, -2.f, 4.f, -4.f};
    copy_data(a, input);
    auto result = backend->create_tensor(element::f32, shape);
    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_TRUE(test::all_close_f(vector<float>{0.00000000f,
                                                0.24740396f,
                                                -0.24740396f,
                                                0.47942554f,
                                                -0.47942554f,
                                                0.84147098f,
                                                -0.84147098f,
                                                0.90929743f,
                                                -0.90929743f,
                                                -0.75680250f,
                                                0.75680250f},
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, sinh)
{
    Shape shape{6};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sinh>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 5.0f, -5.0f};
    copy_data(a, input);
    auto result = backend->create_tensor(element::f32, shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return sinhf(x); });

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_TRUE(test::all_close_f(input, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, sqrt)
{
    Shape shape{2, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sqrt>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{16, 4, 81, 100, 10000, 0});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_TRUE(test::all_close_f(vector<float>{4, 2, 9, 10, 100, 0}, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, tan)
{
    Shape shape{11};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Tan>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> input{0.f, 0.25f, -0.25f, 0.5f, -0.5f, 1.f, -1.f, 2.f, -2.f, 4.f, -4.f};
    copy_data(a, input);
    auto result = backend->create_tensor(element::f32, shape);
    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_TRUE(test::all_close_f(vector<float>{0.00000000f,
                                                0.25534192f,
                                                -0.25534192f,
                                                0.54630249f,
                                                -0.54630249f,
                                                1.55740772f,
                                                -1.55740772f,
                                                -2.18503986f,
                                                2.18503986f,
                                                1.15782128f,
                                                -1.15782128f},
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, tanh)
{
    Shape shape{6};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Tanh>(A), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 0.5f, -0.5f};
    copy_data(a, input);
    auto result = backend->create_tensor(element::f32, shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return tanhf(x); });

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_TRUE(test::all_close_f(input, read_vector<float>(result)));
}
