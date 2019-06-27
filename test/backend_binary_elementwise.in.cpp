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

NGRAPH_TEST(${BACKEND_NAME}, add)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Add>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f(read_vector<float>(result),
                                  (test::NDArray<float, 2>({{6, 8}, {10, 12}})).get_vector()));
}

NGRAPH_TEST(${BACKEND_NAME}, add_overload)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(A + B, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f(read_vector<float>(result),
                                  (test::NDArray<float, 2>({{6, 8}, {10, 12}})).get_vector()));
}

NGRAPH_TEST(${BACKEND_NAME}, multiply)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Multiply>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f(read_vector<float>(result),
                                  (test::NDArray<float, 2>({{5, 12}, {21, 32}})).get_vector()));
}

NGRAPH_TEST(${BACKEND_NAME}, multiply_overload)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(A * B, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> b = backend->create_tensor(element::f32, shape);
    shared_ptr<runtime::Tensor> result = backend->create_tensor(element::f32, shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f(read_vector<float>(result),
                                  (test::NDArray<float, 2>({{5, 12}, {21, 32}})).get_vector()));
}

NGRAPH_TEST(${BACKEND_NAME}, divide)
{
    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Divide>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{2, 2, 2, 2}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, divide_int32)
{
    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::Divide>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{0x40000140, 0x40000001, 8, 16});
    auto b = backend->create_tensor(element::i32, shape);
    copy_data(b, vector<int32_t>{2, 5, 4, 8});
    auto result = backend->create_tensor(element::i32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ((vector<int32_t>{536871072, 214748365, 2, 2}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, divide_cpp_rounding_int32)
{
    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::Divide>(A, B, false), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{-10, -10, 10, 10});
    auto b = backend->create_tensor(element::i32, shape);
    copy_data(b, vector<int32_t>{-3, 3, -3, 3});
    auto result = backend->create_tensor(element::i32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ((vector<int32_t>{3, -3, -3, 3}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, divide_python_rounding_int32)
{
    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::Divide>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{-10, -10, 10, 10});
    auto b = backend->create_tensor(element::i32, shape);
    copy_data(b, vector<int32_t>{-3, 3, -3, 3});
    auto result = backend->create_tensor(element::i32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ((vector<int32_t>{3, -4, -4, 3}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, divide_overload)
{
    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(A / B, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{2, 2, 2, 2}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, divide_adjoint_stability)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape{2, 2};

    auto make_external = [&]() {
        auto A = make_shared<op::Parameter>(element::f32, shape);
        auto B = make_shared<op::Parameter>(element::f32, shape);
        auto f = make_shared<Function>(make_shared<op::Divide>(A, B), ParameterVector{A, B});

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

    auto handle = backend->compile(bf);
    handle->call_with_validate({resulta, resultb}, {a, b, c});
    EXPECT_TRUE(
        test::all_close_f((vector<float>{0.5, 0.5, 0.5, 0.5}), read_vector<float>(resulta)));
    EXPECT_TRUE(
        test::all_close_f((vector<float>{-0.0, -0.0, -0.25, -0.25}), read_vector<float>(resultb)));
}

NGRAPH_TEST(${BACKEND_NAME}, divide_by_zero_float32)
{
    Shape shape{2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Divide>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{0, 0, 0, 0});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ((vector<float>{std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity()}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, maximum)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Maximum>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0.5, 2, 1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8, 0, 0, 1, 1.5});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{1, 8, 4, 17, 0, 0.5, 2, 1.5}),
                                  read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, minimum)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Minimum>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0.5, 2, 1});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8, 0, 0, 1, 1.5});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(
        test::all_close_f((vector<float>{1, 2, -8, 8, -.5, 0, 1, 1}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, minimum_int32)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::Minimum>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{1, 8, -8, 17, -5, 67635216, 2, 1});
    auto b = backend->create_tensor(element::i32, shape);
    copy_data(b, vector<int32_t>{1, 2, 4, 8, 0, 18448, 1, 6});
    auto result = backend->create_tensor(element::i32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ((vector<int32_t>{1, 2, -8, 8, -5, 18448, 1, 1}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, minimum_int64)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::i64, shape);
    auto B = make_shared<op::Parameter>(element::i64, shape);
    auto f = make_shared<Function>(make_shared<op::Minimum>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i64, shape);
    copy_data(a, vector<int64_t>{1, 8, -8, 17, -5, 67635216, 2, 17179887632});
    auto b = backend->create_tensor(element::i64, shape);
    copy_data(b, vector<int64_t>{1, 2, 4, 8, 0, 18448, 1, 280592});
    auto result = backend->create_tensor(element::i64, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ((vector<int64_t>{1, 2, -8, 8, -5, 18448, 1, 280592}), read_vector<int64_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, maximum_int32)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::Maximum>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape);
    copy_data(a, vector<int32_t>{0x40000140, 0x40000001, -8, 17});
    auto b = backend->create_tensor(element::i32, shape);
    copy_data(b, vector<int32_t>{0x40000170, 0x40000000, 4, 8});
    auto result = backend->create_tensor(element::i32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ((vector<int32_t>{0x40000170, 0x40000001, 4, 17}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, maximum_int64)
{
    Shape shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::i64, shape);
    auto B = make_shared<op::Parameter>(element::i64, shape);
    auto f = make_shared<Function>(make_shared<op::Maximum>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i64, shape);
    copy_data(a, vector<int64_t>{1, 8, -8, 17, -5, 67635216, 2, 17179887632});
    auto b = backend->create_tensor(element::i64, shape);
    copy_data(b, vector<int64_t>{1, 2, 4, 8, 0, 18448, 1, 280592});
    auto result = backend->create_tensor(element::i64, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_EQ((vector<int64_t>{1, 8, 4, 17, 0, 67635216, 2, 17179887632}),
              read_vector<int64_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, power)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Power>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 5});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{2, 0, 6, 3});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f(vector<float>{1, 1, 729, 125}, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, subtract)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Subtract>(A, B), ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{1, 2, 4, 8}), read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, subtract_overload)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(A - B, ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->create_tensor(element::f32, shape);
    copy_data(b, vector<float>{1, 2, 4, 8});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close_f((vector<float>{1, 2, 4, 8}), read_vector<float>(result)));
}

template <typename optype, typename itype, typename otype>
void check_auto_bcast(const std::vector<std::vector<itype>>& inputs,
                      const std::vector<otype> output)
{
    auto iet = element::from<itype>();
    auto oet = element::from<otype>();

    if (std::is_same<itype, char>::value)
    {
        iet = element::boolean;
    }
    if (std::is_same<otype, char>::value)
    {
        oet = element::boolean;
    }
    auto A = make_shared<op::Parameter>(iet, Shape{2, 3});
    auto B = make_shared<op::Parameter>(iet, Shape{3});
    auto f = make_shared<Function>(make_shared<optype>(A, B, op::AutoBroadcastType::NUMPY),
                                   ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    shared_ptr<runtime::Tensor> a = backend->create_tensor(iet, Shape{2, 3});
    shared_ptr<runtime::Tensor> b = backend->create_tensor(iet, Shape{3});
    shared_ptr<runtime::Tensor> result = backend->create_tensor(oet, Shape{2, 3});

    copy_data(a, inputs[0]);
    copy_data(b, inputs[1]);

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a, b});
    EXPECT_TRUE(test::all_close(read_vector<otype>(result), output));
}

NGRAPH_TEST(${BACKEND_NAME}, auto_bcast_binary_elementwise)
{
    check_auto_bcast<op::Add, float, float>({{1, 2, 3, 4, 5, 6}, {5, 6, 7}}, {6, 8, 10, 9, 11, 13});
    check_auto_bcast<op::Subtract, float, float>({{1, 2, 3, 4, 5, 6}, {5, 6, 7}},
                                                 {-4.f, -4.f, -4.f, -1.f, -1.f, -1.f});
    check_auto_bcast<op::Multiply, float, float>({{1, 2, 3, 4, 5, 6}, {5, 6, 7}},
                                                 {5, 12, 21, 20, 30, 42});
    check_auto_bcast<op::Divide, float, float>({{4, 5, 6, 7, 8, 9}, {1, 2, 3}},
                                               {4, 2.5f, 2, 7, 4, 3});
    check_auto_bcast<op::Maximum, float, float>({{1, 2, 3, 4, 5, 6}, {1, 5, 8}},
                                                {1, 5, 8, 4, 5, 8});
    check_auto_bcast<op::Minimum, float, float>({{1, 2, 3, 4, 5, 6}, {1, 5, 8}},
                                                {1, 2, 3, 1, 5, 6});
    check_auto_bcast<op::Power, float, float>({{1, 2, 3, 4, 5, 6}, {1, 2, 3}},
                                              {1, 4, 27, 4, 25, 216});

    check_auto_bcast<op::And, char, char>({{1, 0, 1, 0, 0, 1}, {1, 0, 1}}, {1, 0, 1, 0, 0, 1});
    check_auto_bcast<op::Or, char, char>({{1, 0, 1, 0, 1, 1}, {1, 0, 0}}, {1, 0, 1, 1, 1, 1});

    check_auto_bcast<op::Equal, uint8_t, char>({{1, 0, 1, 0, 1, 1}, {1, 0, 0}}, {1, 1, 0, 0, 0, 0});
    check_auto_bcast<op::Greater, float, char>({{1, 2, 3, 4, 5, 6}, {1, 5, 8}}, {0, 0, 0, 1, 0, 0});
    check_auto_bcast<op::GreaterEq, float, char>({{1, 2, 3, 4, 5, 6}, {1, 5, 8}},
                                                 {1, 0, 0, 1, 1, 0});
    check_auto_bcast<op::Less, uint8_t, char>({{1, 2, 3, 4, 5, 6}, {1, 5, 8}}, {0, 1, 1, 0, 0, 1});
    check_auto_bcast<op::LessEq, uint8_t, char>({{1, 2, 3, 4, 5, 6}, {1, 5, 8}},
                                                {1, 1, 1, 0, 1, 1});
    check_auto_bcast<op::NotEqual, uint8_t, char>({{1, 2, 3, 4, 5, 6}, {1, 5, 8}},
                                                  {0, 1, 1, 1, 0, 1});
}
