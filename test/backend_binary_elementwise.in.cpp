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

NGRAPH_TEST(${BACKEND_NAME}, add)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Add>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{6, 8}, {10, 12}})).get_vector());
}

NGRAPH_TEST(${BACKEND_NAME}, add_overload)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(A + B, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{6, 8}, {10, 12}})).get_vector());
}

NGRAPH_TEST(${BACKEND_NAME}, multiply)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Multiply>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{5, 12}, {21, 32}})).get_vector());
}

NGRAPH_TEST(${BACKEND_NAME}, multiply_overload)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(A * B, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ(read_vector<float>(result),
              (test::NDArray<float, 2>({{5, 12}, {21, 32}})).get_vector());
}

NGRAPH_TEST(${BACKEND_NAME}, divide)
{
    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Divide>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{2, 2, 2, 2}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, divide_overload)
{
    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(A / B, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{2, 2, 2, 2}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, divide_adjoint_stability)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{2, 2};

    auto make_external = [&]() {
        auto A = make_shared<op::Parameter>(element::f32, shape);
        auto B = make_shared<op::Parameter>(element::f32, shape);
        auto f = make_shared<Function>(make_shared<op::Divide>(A, B), op::ParameterVector{A, B});

        auto Y_out = f->get_output_op(0);
        auto Xs = f->get_parameters();
        auto C = std::make_shared<op::Parameter>(Y_out->get_element_type(), Y_out->get_shape());
        ngraph::autodiff::Adjoints adjoints(NodeVector{Y_out}, NodeVector{C});
        std::vector<std::shared_ptr<Node>> dYdXs(Xs.size());
        transform(
            Xs.begin(), Xs.end(), dYdXs.begin(), [C, &adjoints](const std::shared_ptr<Node>& X) {
                return adjoints.backprop_node(X);
            });
        std::vector<std::shared_ptr<op::Parameter>> params(Xs);
        params.push_back(C);

        auto bf = std::make_shared<Function>(dYdXs, params);

        return bf;
    };

    auto bf = make_external();

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{0, 0, 1, 1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{2, 2, 2, 2});
    auto c = backend->create_tensor(element::f32, shape);
    copy_data(c, vector<float>{1, 1, 1, 1});

    auto resulta = backend->create_tensor(element::f32, shape);
    auto resultb = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(bf, {resulta, resultb}, {a, b, c});
    EXPECT_EQ((vector<float>{0.5, 0.5, 0.5, 0.5}), read_vector<float>(resulta));
    EXPECT_EQ((vector<float>{-0.0, -0.0, -0.25, -0.25}), read_vector<float>(resultb));
}

NGRAPH_TEST(${BACKEND_NAME}, divide_by_zero_float32)
{
    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Divide>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{0, 0, 0, 0});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity()}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, divide_by_zero_int32)
{
    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::Divide>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::i32, shape);
    copy_data(b, vector<int>{0, 0, 0, 0});
    auto result = backend->create_tensor(element::i32, shape);

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wused-but-marked-unused"
#pragma clang diagnostic ignored "-Wcovered-switch-default"

    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    EXPECT_DEATH_IF_SUPPORTED(
        {
            try
            {
                backend->call_with_validate(f, {result}, {a, b});
            }
            catch (...)
            {
                abort();
            }
        },
        "");

#pragma clang diagnostic pop
}

NGRAPH_TEST(${BACKEND_NAME}, maximum)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Maximum>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0.5, 2, 1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8, 0, 0, 1, 1.5});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{1, 8, 4, 17, 0, 0.5, 2, 1.5}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, minimum)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Minimum>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0.5, 2, 1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8, 0, 0, 1, 1.5});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{1, 2, -8, 8, -.5, 0, 1, 1}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, minimum_int32)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::Minimum>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{1, 8, -8, 17, -5, 67635216, 2, 1});
    auto b = backend->create_tensor(element::i32, shape);
    copy_data(b, vector<int32_t>{1, 2, 4, 8, 0, 18448, 1, 6});
    auto result = backend->create_tensor(element::i32, shape);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<int32_t>{1, 2, -8, 8, -5, 18448, 1, 1}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, minimum_int64)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::i64, shape);
    auto B = make_shared<op::Parameter>(element::i64, shape);
    auto f = make_shared<Function>(make_shared<op::Minimum>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i64, shape);
    copy_data(a, vector<int64_t>{1, 8, -8, 17, -5, 67635216, 2, 17179887632});
    auto b = backend->create_tensor(element::i64, shape);
    copy_data(b, vector<int64_t>{1, 2, 4, 8, 0, 18448, 1, 280592});
    auto result = backend->create_tensor(element::i64, shape);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<int64_t>{1, 2, -8, 8, -5, 18448, 1, 280592}), read_vector<int64_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, maximum_int32)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::Maximum>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{1, 8, -8, 17, -5, 67635216, 2, 1});
    auto b = backend->create_tensor(element::i32, shape);
    copy_data(b, vector<int32_t>{1, 2, 4, 8, 0, 18448, 1, 6});
    auto result = backend->create_tensor(element::i32, shape);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<int32_t>{1, 8, 4, 17, 0, 67635216, 2, 6}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, maximum_int64)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::i64, shape);
    auto B = make_shared<op::Parameter>(element::i64, shape);
    auto f = make_shared<Function>(make_shared<op::Maximum>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i64, shape);
    copy_data(a, vector<int64_t>{1, 8, -8, 17, -5, 67635216, 2, 17179887632});
    auto b = backend->create_tensor(element::i64, shape);
    copy_data(b, vector<int64_t>{1, 2, 4, 8, 0, 18448, 1, 280592});
    auto result = backend->create_tensor(element::i64, shape);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<int64_t>{1, 8, 4, 17, 0, 67635216, 2, 17179887632}),
              read_vector<int64_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, power)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Power>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 5});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{2, 0, 6, 3});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_TRUE(test::all_close(vector<float>{1, 1, 729, 125}, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, subtract)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Subtract>(A, B), op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{1, 2, 4, 8}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, subtract_overload)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(A - B, op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((vector<float>{1, 2, 4, 8}), read_vector<float>(result));
}
