// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <algorithm>
#include "gtest/gtest.h"

#include <cmath>
#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;

TEST(execute, abc)
{
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto C = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>((A + B) * C, rt, op::Parameters{A, B, C});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *a = vector<float>{1, 2, 3, 4};
    auto b = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *b = vector<float>{5, 6, 7, 8};
    auto c = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *c = vector<float>{9, 10, 11, 12};
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape);

    (*cf)({a, b, c}, {result});
    ASSERT_EQ((vector<float>{54, 80, 110, 144}), result->get_vector());

    (*cf)({b, a, c}, {result});
    ASSERT_EQ((vector<float>{54, 80, 110, 144}), result->get_vector());

    (*cf)({a, c, b}, {result});
    ASSERT_EQ((vector<float>{50, 72, 98, 128}), result->get_vector());
}

TEST(execute, abc_int64)
{
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Int64::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Int64::element_type(), shape);
    auto C = make_shared<op::Parameter>(element::Int64::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Int64::element_type(), shape);
    auto f = make_shared<Function>((A + B) * C, rt, op::Parameters{A, B, C});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Int64>(shape);
    *a = vector<element::Int64::type>{1, 2, 3, 4};
    auto b = backend->make_parameterized_tensor_view<element::Int64>(shape);
    *b = vector<element::Int64::type>{5, 6, 7, 8};
    auto c = backend->make_parameterized_tensor_view<element::Int64>(shape);
    *c = vector<element::Int64::type>{9, 10, 11, 12};
    auto result = backend->make_parameterized_tensor_view<element::Int64>(shape);

    (*cf)({a, b, c}, {result});
    ASSERT_EQ((vector<element::Int64::type>{54, 80, 110, 144}), result->get_vector());

    (*cf)({b, a, c}, {result});
    ASSERT_EQ((vector<element::Int64::type>{54, 80, 110, 144}), result->get_vector());

    (*cf)({a, c, b}, {result});
    ASSERT_EQ((vector<element::Int64::type>{50, 72, 98, 128}), result->get_vector());
}

// Same as abc, but using tuples for input and output
TEST(execute, abc_tuple)
{
    auto shape = Shape{2, 2};

    auto tensor_view_type = make_shared<TensorViewType>(element::Float32::element_type(), shape);

    auto ABC = make_shared<op::Parameter>(
        make_shared<TupleType>(ValueTypes{tensor_view_type, tensor_view_type, tensor_view_type}));

    auto A = make_shared<op::GetTupleElement>(ABC, 0);
    auto B = make_shared<op::GetTupleElement>(ABC, 1);
    auto C = make_shared<op::GetTupleElement>(ABC, 2);
    auto f = make_shared<Function>(
        make_shared<op::Tuple>(Nodes{(A + B) * C}), tensor_view_type, op::Parameters{ABC});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *a = vector<float>{1, 2, 3, 4};
    auto b = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *b = vector<float>{5, 6, 7, 8};
    auto c = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *c = vector<float>{9, 10, 11, 12};
    auto abc = ngraph::runtime::make_tuple({a, b, c});
    auto bac = ngraph::runtime::make_tuple({b, a, c});
    auto acb = ngraph::runtime::make_tuple({a, c, b});
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape);
    auto result_tuple = ngraph::runtime::make_tuple({result});

    (*cf)({abc}, {result_tuple});
    ASSERT_EQ((vector<float>{54, 80, 110, 144}), result->get_vector());

    (*cf)({bac}, {result_tuple});
    ASSERT_EQ((vector<float>{54, 80, 110, 144}), result->get_vector());

    (*cf)({acb}, {result_tuple});
    ASSERT_EQ((vector<float>{50, 72, 98, 128}), result->get_vector());
}

// Same as abc, but using tuples for input and output
TEST(execute, abc_tuple_int64)
{
    auto shape = Shape{2, 2};

    auto tensor_view_type = make_shared<TensorViewType>(element::Int64::element_type(), shape);

    auto ABC = make_shared<op::Parameter>(
        make_shared<TupleType>(ValueTypes{tensor_view_type, tensor_view_type, tensor_view_type}));

    auto A = make_shared<op::GetTupleElement>(ABC, 0);
    auto B = make_shared<op::GetTupleElement>(ABC, 1);
    auto C = make_shared<op::GetTupleElement>(ABC, 2);
    auto f = make_shared<Function>(
        make_shared<op::Tuple>(Nodes{(A + B) * C}), tensor_view_type, op::Parameters{ABC});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Int64>(shape);
    *a = vector<element::Int64::type>{1, 2, 3, 4};
    auto b = backend->make_parameterized_tensor_view<element::Int64>(shape);
    *b = vector<element::Int64::type>{5, 6, 7, 8};
    auto c = backend->make_parameterized_tensor_view<element::Int64>(shape);
    *c = vector<element::Int64::type>{9, 10, 11, 12};
    auto abc = ngraph::runtime::make_tuple({a, b, c});
    auto bac = ngraph::runtime::make_tuple({b, a, c});
    auto acb = ngraph::runtime::make_tuple({a, c, b});
    auto result = ngraph::runtime::make_tensor<element::Int64>(shape);
    auto result_tuple = ngraph::runtime::make_tuple({result});

    (*cf)({abc}, {result_tuple});
    ASSERT_EQ((vector<element::Int64::type>{54, 80, 110, 144}), result->get_vector());

    (*cf)({bac}, {result_tuple});
    ASSERT_EQ((vector<element::Int64::type>{54, 80, 110, 144}), result->get_vector());

    (*cf)({acb}, {result_tuple});
    ASSERT_EQ((vector<element::Int64::type>{50, 72, 98, 128}), result->get_vector());
}

// Multiple retrive values
TEST(execute, tuple_result)
{
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto C = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto A_add_B = make_shared<op::Add>(A, B);
    auto A_add_B_mul_C = make_shared<op::Multiply>(A_add_B, C);

    auto rt = make_shared<TupleType>(std::vector<shared_ptr<const ValueType>>(
        {make_shared<TensorViewType>(element::Float32::element_type(), shape),
         make_shared<TensorViewType>(element::Float32::element_type(), shape)}));
    auto f = make_shared<Function>(
        make_shared<op::Tuple>(Nodes{A_add_B, A_add_B_mul_C}), rt, op::Parameters{A, B, C});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *a = vector<float>{1, 2, 3, 4};
    auto b = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *b = vector<float>{5, 6, 7, 8};
    auto c = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *c = vector<float>{9, 10, 11, 12};

    auto r0 = backend->make_parameterized_tensor_view<element::Float32>(shape);
    auto r1 = backend->make_parameterized_tensor_view<element::Float32>(shape);
    auto result_tuple = ngraph::runtime::make_tuple({r0, r1});

    (*cf)({a, b, c}, {result_tuple});

    ASSERT_EQ((vector<float>{6, 8, 10, 12}), r0->get_vector());
    ASSERT_EQ((vector<float>{54, 80, 110, 144}), r1->get_vector());
}

TEST(execute, abs)
{
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto result_type = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Abs>(A), result_type, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *a = vector<float>{1, -2, 0, -4.8f};
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape);

    (*cf)({a}, {result});
    ASSERT_EQ((vector<float>{1, 2, 0, 4.8f}), result->get_vector());
}

TEST(execute, concat_matrix_colwise)
{
    auto shape_a = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_b = Shape{2, 3};
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape_b);
    auto shape_c = Shape{2, 3};
    auto C = make_shared<op::Parameter>(element::Float32::element_type(), shape_c);
    auto shape_r = Shape{2, 8};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), Shape{2, 8});
    auto f = make_shared<Function>(
        make_shared<op::Concat>(Nodes{A, B, C}, 1), rt, op::Parameters{A, B, C});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape_a);
    *a = vector<float>{2, 4, 8, 16};
    auto b = backend->make_parameterized_tensor_view<element::Float32>(shape_b);
    *b = vector<float>{1, 2, 4, 8, 16, 32};
    auto c = backend->make_parameterized_tensor_view<element::Float32>(shape_c);
    *c = vector<float>{2, 3, 5, 7, 11, 13};
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape_r);

    (*cf)({a, b, c}, {result});
    ASSERT_EQ((vector<float>{2, 4, 1, 2, 4, 2, 3, 5, 8, 16, 8, 16, 32, 7, 11, 13}),
              result->get_vector());
}

TEST(execute, concat_matrix_rowwise)
{
    auto shape_a = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_b = Shape{3, 2};
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape_b);
    auto shape_c = Shape{3, 2};
    auto C = make_shared<op::Parameter>(element::Float32::element_type(), shape_c);
    auto shape_r = Shape{8, 2};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), Shape{8, 2});
    auto f = make_shared<Function>(
        make_shared<op::Concat>(Nodes{A, B, C}, 0), rt, op::Parameters{A, B, C});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape_a);
    *a = vector<float>{2, 4, 8, 16};
    auto b = backend->make_parameterized_tensor_view<element::Float32>(shape_b);
    *b = vector<float>{1, 2, 4, 8, 16, 32};
    auto c = backend->make_parameterized_tensor_view<element::Float32>(shape_c);
    *c = vector<float>{2, 3, 5, 7, 11, 13};
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape_r);

    (*cf)({a, b, c}, {result});
    ASSERT_EQ((vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13}),
              result->get_vector());
}

TEST(execute, concat_matrix_int64)
{
    auto shape_a = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Int64::element_type(), shape_a);
    auto shape_b = Shape{3, 2};
    auto B = make_shared<op::Parameter>(element::Int64::element_type(), shape_b);
    auto shape_c = Shape{3, 2};
    auto C = make_shared<op::Parameter>(element::Int64::element_type(), shape_c);
    auto shape_r = Shape{8, 2};
    auto rt = make_shared<TensorViewType>(element::Int64::element_type(), Shape{8, 2});
    auto f = make_shared<Function>(
        make_shared<op::Concat>(Nodes{A, B, C}, 0), rt, op::Parameters{A, B, C});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Int64>(shape_a);
    *a = vector<element::Int64::type>{2, 4, 8, 16};
    auto b = backend->make_parameterized_tensor_view<element::Int64>(shape_b);
    *b = vector<element::Int64::type>{1, 2, 4, 8, 16, 32};
    auto c = backend->make_parameterized_tensor_view<element::Int64>(shape_c);
    *c = vector<element::Int64::type>{2, 3, 5, 7, 11, 13};
    auto result = backend->make_parameterized_tensor_view<element::Int64>(shape_r);

    (*cf)({a, b, c}, {result});
    ASSERT_EQ((vector<element::Int64::type>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13}),
              result->get_vector());
}

TEST(execute, concat_vector)
{
    auto shape_a = Shape{4};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_b = Shape{6};
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape_b);
    auto shape_c = Shape{2};
    auto C = make_shared<op::Parameter>(element::Float32::element_type(), shape_c);
    auto shape_r = Shape{12};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), Shape{12});
    auto f = make_shared<Function>(
        make_shared<op::Concat>(Nodes{A, B, C}, 0), rt, op::Parameters{A, B, C});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape_a);
    *a = vector<float>{2, 4, 8, 16};
    auto b = backend->make_parameterized_tensor_view<element::Float32>(shape_b);
    *b = vector<float>{1, 2, 4, 8, 16, 32};
    auto c = backend->make_parameterized_tensor_view<element::Float32>(shape_c);
    *c = vector<float>{18, 19};
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape_r);

    (*cf)({a, b, c}, {result});
    ASSERT_EQ((vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 18, 19}), result->get_vector());
}

TEST(execute, divide)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    auto shape = Shape{2, 2};

    auto make_external = [&]() {
        auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
        auto f = make_shared<Function>(make_shared<op::Divide>(A, B), rt, op::Parameters{A, B});

        auto external = manager->compile(f);
        return external;
    };

    auto cf = backend->make_call_frame(make_external());

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *a = vector<float>{2, 4, 8, 16};
    auto b = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *b = vector<float>{1, 2, 4, 8};
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<float>{2, 2, 2, 2}), result->get_vector());
}

TEST(execute, equal)
{
    auto shape = Shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Bool::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Equal>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *a = vector<float>{1, 8, -8, 17, -0.5, 0, 1, 1};
    auto b = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *b = vector<float>{1, 8, 4, 8, 0, 0, 1, 1.5};
    auto result = backend->make_parameterized_tensor_view<element::Bool>(shape);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<char>{1, 1, 0, 0, 0, 1, 1, 0}), result->get_vector());
}

TEST(execute, dot_0_0)
{
    auto shape = Shape{0};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto shape_r = Shape{};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), Shape{});
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *a = vector<float>{};
    auto b = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *b = vector<float>{};
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape_r);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<float>{0}), result->get_vector());
}

TEST(execute, dot_matrix_2x0_0x2)
{
    auto shape_a = Shape{2, 0};
    auto shape_b = Shape{0, 2};
    auto shape_r = Shape{2, 2};

    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    auto make_external = [&]() {
        auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
        auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape_b);
        auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
        auto f = make_shared<Function>(make_shared<op::Dot>(A, B), rt, op::Parameters{A, B});

        auto external = manager->compile(f);
        return external;
    };

    auto cf = backend->make_call_frame(make_external());

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape_a);
    *a = vector<float>{};
    auto b = backend->make_parameterized_tensor_view<element::Float32>(shape_b);
    *b = vector<float>{};
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape_r);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<float>{0, 0, 0, 0}), result->get_vector());
}

TEST(execute, dot_matrix_0x2_2x0)
{
    auto shape_a = Shape{0, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_b = Shape{2, 0};
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape_b);
    auto shape_r = Shape{0, 0};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape_a);
    *a = vector<float>{};
    auto b = backend->make_parameterized_tensor_view<element::Float32>(shape_b);
    *b = vector<float>{};
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape_r);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<float>{}), result->get_vector());
}

TEST(execute, dot_matrix_3x2_2x0)
{
    auto shape_a = Shape{3, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_b = Shape{2, 0};
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape_b);
    auto shape_r = Shape{0, 0};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape_a);
    *a = vector<float>{1, 2, 3, 4, 5, 6};
    auto b = backend->make_parameterized_tensor_view<element::Float32>(shape_b);
    *b = vector<float>{};
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape_r);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<float>{}), result->get_vector());
}

TEST(execute, dot_scalar_0x2)
{
    auto shape_a = Shape{};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_b = Shape{0, 2};
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape_b);
    auto shape_r = Shape{0, 2};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape_a);
    *a = vector<float>{1};
    auto b = backend->make_parameterized_tensor_view<element::Float32>(shape_b);
    *b = vector<float>{};
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape_r);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<float>{}), result->get_vector());
}

TEST(execute, dot_2x0_0)
{
    auto shape_a = Shape{2, 0};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_b = Shape{0};
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape_b);
    auto shape_r = Shape{2};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape_a);
    *a = vector<float>{};
    auto b = backend->make_parameterized_tensor_view<element::Float32>(shape_b);
    *b = vector<float>{};
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape_r);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<float>{0, 0}), result->get_vector());
}

TEST(execute, dot1d)
{
    auto shape = Shape{4};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto shape_r = Shape{1};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), Shape{});
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *a = vector<float>{2, 4, 8, 16};
    auto b = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *b = vector<float>{1, 2, 4, 8};
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape_r);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<float>{170}), result->get_vector());
}

TEST(execute, dot2d)
{
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto shape_r = Shape{2, 2};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *a = vector<float>{1, 2, 3, 4};
    auto b = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *b = vector<float>{5, 6, 7, 8};
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape_r);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<float>{19, 22, 43, 50}), result->get_vector());
}

TEST(execute, dot_scalar_tensor_arg0)
{
    auto shape_a = Shape{};
    auto shape_b = Shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape_b);
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_b);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape_a);
    *a = vector<float>{6};
    auto b = backend->make_parameterized_tensor_view<element::Float32>(shape_b);
    *b = vector<float>{1, 2, 3, 4, 5, 6, 7, 8};
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape_b);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<float>{6, 12, 18, 24, 30, 36, 42, 48}), result->get_vector());
}

TEST(execute, dot_scalar_tensor_arg1)
{
    auto shape_a = Shape{2, 2, 2};
    auto shape_b = Shape{};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape_b);
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_a);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape_a);
    *a = vector<float>{1, 2, 3, 4, 5, 6, 7, 8};
    auto b = backend->make_parameterized_tensor_view<element::Float32>(shape_b);
    *b = vector<float>{6};
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape_a);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<float>{6, 12, 18, 24, 30, 36, 42, 48}), result->get_vector());
}

TEST(execute, dot_scalar_scalar)
{
    auto shape = Shape{};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *a = vector<float>{8};
    auto b = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *b = vector<float>{6};
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<float>{48}), result->get_vector());
}

TEST(execute, dot_matrix_vector)
{
    auto shape_a = Shape{4, 4};
    auto shape_b = Shape{4};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape_b);
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_b);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), rt, op::Parameters{A, B});
    auto shape_r = Shape{4};

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape_a);
    *a = vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    auto b = backend->make_parameterized_tensor_view<element::Float32>(shape_b);
    *b = vector<float>{17, 18, 19, 20};
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape_r);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<float>{190, 486, 782, 1078}), result->get_vector());
}

TEST(execute, dot_matrix_vector_int64)
{
    auto shape_a = Shape{4, 4};
    auto shape_b = Shape{4};
    auto A = make_shared<op::Parameter>(element::Int64::element_type(), shape_a);
    auto B = make_shared<op::Parameter>(element::Int64::element_type(), shape_b);
    auto rt = make_shared<TensorViewType>(element::Int64::element_type(), shape_b);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), rt, op::Parameters{A, B});
    auto shape_r = Shape{4};

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Int64>(shape_a);
    *a = vector<element::Int64::type>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    auto b = backend->make_parameterized_tensor_view<element::Int64>(shape_b);
    *b = vector<element::Int64::type>{17, 18, 19, 20};
    auto result = backend->make_parameterized_tensor_view<element::Int64>(shape_r);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<element::Int64::type>{190, 486, 782, 1078}), result->get_vector());
}

TEST(execute, greater)
{
    auto shape = Shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Bool::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Greater>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *a = vector<float>{1, 8, -8, 17, -0.5, 0.5, 2, 1};
    auto b = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *b = vector<float>{1, 2, 4, 8, 0, 0, 1, 1.5};
    auto result = backend->make_parameterized_tensor_view<element::Bool>(shape);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<char>{0, 1, 0, 1, 0, 1, 1, 0}), result->get_vector());
}

TEST(execute, greatereq)
{
    auto shape = Shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Bool::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::GreaterEq>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *a = vector<float>{1, 8, -8, 17, -0.5, 0, 2, 1};
    auto b = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *b = vector<float>{1, 2, -8, 8, 0, 0, 0.5, 1.5};
    auto result = backend->make_parameterized_tensor_view<element::Bool>(shape);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<char>{1, 1, 1, 1, 0, 1, 1, 0}), result->get_vector());
}

TEST(execute, less)
{
    auto shape = Shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Bool::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Less>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *a = vector<float>{1, 8, -8, 17, -0.5, 0.5, 2, 1};
    auto b = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *b = vector<float>{1, 2, 4, 8, 0, 0, 1, 1.5};
    auto result = backend->make_parameterized_tensor_view<element::Bool>(shape);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<char>{0, 0, 1, 0, 1, 0, 0, 1}), result->get_vector());
}

TEST(execute, lesseq)
{
    auto shape = Shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Bool::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::LessEq>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *a = vector<float>{1, 8, -8, 17, -0.5, 0, 2, 1};
    auto b = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *b = vector<float>{1, 2, -8, 8, 0, 0, 0.5, 1.5};
    auto result = backend->make_parameterized_tensor_view<element::Bool>(shape);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<char>{1, 0, 1, 0, 1, 1, 0, 1}), result->get_vector());
}

TEST(execute, log)
{
    auto shape = Shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Log>(A), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *a = vector<float>{expf(1), expf(2), expf(3), expf(4), expf(5), expf(6), expf(7), expf(8)};
    vector<float> loga;
    for (auto elt : a->get_vector())
    {
        loga.push_back(logf(elt));
    }
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape);

    (*cf)({a}, {result});
    ASSERT_EQ(loga, result->get_vector());
}

TEST(execute, maximum)
{
    auto shape = Shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Maximum>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *a = vector<float>{1, 8, -8, 17, -0.5, 0.5, 2, 1};
    auto b = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *b = vector<float>{1, 2, 4, 8, 0, 0, 1, 1.5};
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<float>{1, 8, 4, 17, 0, 0.5, 2, 1.5}), result->get_vector());
}

TEST(execute, negative)
{
    auto shape = Shape{2, 3};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Negative>(A), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *a = vector<float>{1, -2, 0, -4.8f, 8.6f, -8.6f};
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape);

    (*cf)({a}, {result});
    ASSERT_EQ((vector<float>{-1, 2, 0, 4.8f, -8.6f, 8.6f}), result->get_vector());
}

TEST(execute, notequal)
{
    auto shape = Shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Bool::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::NotEqual>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *a = vector<float>{1, 8, -8, 17, -0.5, 0, 1, 1};
    auto b = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *b = vector<float>{1, 8, 4, 8, 0, 0, 1, 1.5};
    auto result = backend->make_parameterized_tensor_view<element::Bool>(shape);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<char>{0, 0, 1, 1, 1, 0, 0, 1}), result->get_vector());
}

TEST(execute, select)
{
    auto shape = Shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Bool::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto C = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Select>(A, B, C), rt, op::Parameters{A, B, C});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Bool>(shape);
    *a = vector<char>{0, 1, 1, 0, 0, 1, 0, 1};
    auto b = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *b = vector<float>{1, 2, 3, 4, 5, 6, 7, 8};
    auto c = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *c = vector<float>{11, 12, 13, 14, 15, 16, 17, 18};
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape);

    (*cf)({a, b, c}, {result});
    ASSERT_EQ((vector<float>{11, 2, 3, 14, 15, 6, 17, 8}), result->get_vector());
}

TEST(execute, subtract)
{
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Subtract>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *a = vector<float>{2, 4, 8, 16};
    auto b = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *b = vector<float>{1, 2, 4, 8};
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<float>{1, 2, 4, 8}), result->get_vector());
}

TEST(execute, scalar_constant)
{
    auto shape = Shape{};
    auto A = make_shared<op::ScalarConstant<element::Float32>>(-3.0f);
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(A, rt, op::Parameters{});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape);

    (*cf)({}, {result});
    ASSERT_EQ((vector<float>{-3.0f}), result->get_vector());
}

TEST(execute, tensor_constant)
{
    auto shape = Shape{2, 2, 2};
    auto A = make_shared<op::TensorConstant<element::Float32>>(shape);
    A->get_value()->get_vector() = {1, 2, 3, 4, 5, 6, 7, 8};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(A, rt, op::Parameters{});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape);

    (*cf)({}, {result});
    ASSERT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8}), result->get_vector());
}

TEST(execute, tensor_constant_with_op)
{
    auto shape = Shape{2, 2, 2};
    auto A = make_shared<op::TensorConstant<element::Float32>>(shape);
    A->get_value()->get_vector() = {-1, 2, 3, -4, 5, -6, -7, 8};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Abs>(A), rt, op::Parameters{});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape);

    (*cf)({}, {result});
    ASSERT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8}), result->get_vector());
}

TEST(execute, function_call)
{
    // First create "f(A,B,C) = (A+B)*C".
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto C = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt_f = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>((A + B) * C, rt_f, op::Parameters{A, B, C});

    // Now make "g(X,Y,Z) = f(X,Y,Z) + f(X,Y,Z)"
    auto X = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto Y = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto Z = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt_g = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto g = make_shared<Function>(make_shared<op::FunctionCall>(f, Nodes{X, Y, Z}) +
                                       make_shared<op::FunctionCall>(f, Nodes{X, Y, Z}),
                                   rt_g,
                                   op::Parameters{X, Y, Z});

    // Now call g on some test vectors.
    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(g);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    auto x = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *x = vector<float>{1, 2, 3, 4};
    auto y = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *y = vector<float>{5, 6, 7, 8};
    auto z = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *z = vector<float>{9, 10, 11, 12};
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape);

    (*cf)({x, y, z}, {result});
    ASSERT_EQ((vector<float>{108, 160, 220, 288}), result->get_vector());

    (*cf)({y, x, z}, {result});
    ASSERT_EQ((vector<float>{108, 160, 220, 288}), result->get_vector());

    (*cf)({x, z, y}, {result});
    ASSERT_EQ((vector<float>{100, 144, 196, 256}), result->get_vector());
}

TEST(execute, broadcast_scalar_vector)
{
    auto shape_a = Shape{};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_r = Shape{4};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto f = make_shared<Function>(
        make_shared<op::Broadcast>(A, shape_r, AxisSet{0}), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape_a);
    *a = vector<float>{6};
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape_r);

    (*cf)({a}, {result});
    ASSERT_EQ((vector<float>{6, 6, 6, 6}), result->get_vector());
}

TEST(execute, broadcast_scalar_matrix)
{
    auto shape_a = Shape{};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_r = Shape{2, 2};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto f = make_shared<Function>(
        make_shared<op::Broadcast>(A, shape_r, AxisSet{0, 1}), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape_a);
    *a = vector<float>{6};
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape_r);

    (*cf)({a}, {result});
    ASSERT_EQ((vector<float>{6, 6, 6, 6}), result->get_vector());
}

TEST(execute, broadcast_scalar_tensor)
{
    auto shape_a = Shape{};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_r = Shape{2, 2, 2};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto f = make_shared<Function>(
        make_shared<op::Broadcast>(A, shape_r, AxisSet{0, 1, 2}), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape_a);
    *a = vector<float>{6};
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape_r);

    (*cf)({a}, {result});
    ASSERT_EQ((vector<float>{6, 6, 6, 6, 6, 6, 6, 6}), result->get_vector());
}

TEST(execute, broadcast_trivial)
{
    auto shape = Shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(
        make_shared<op::Broadcast>(A, shape, AxisSet{}), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *a = vector<float>{2, 4, 6, 8, 16, 32, 64, 128};
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape);

    (*cf)({a}, {result});
    ASSERT_EQ((vector<float>{2, 4, 6, 8, 16, 32, 64, 128}), result->get_vector());
}

TEST(execute, broadcast_vector_colwise)
{
    auto shape_a = Shape{3};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_r = Shape{3, 4};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto f = make_shared<Function>(
        make_shared<op::Broadcast>(A, shape_r, AxisSet{1}), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape_a);
    *a = vector<float>{1, 2, 3};
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape_r);

    (*cf)({a}, {result});
    ASSERT_EQ((vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}), result->get_vector());
}

TEST(execute, broadcast_vector_rowwise)
{
    auto shape_a = Shape{4};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_r = Shape{3, 4};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto f = make_shared<Function>(
        make_shared<op::Broadcast>(A, shape_r, AxisSet{0}), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape_a);
    *a = vector<float>{1, 2, 3, 4};
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape_r);

    (*cf)({a}, {result});
    ASSERT_EQ((vector<float>{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}), result->get_vector());
}

TEST(execute, broadcast_vector_rowwise_int64)
{
    auto shape_a = Shape{4};
    auto A = make_shared<op::Parameter>(element::Int64::element_type(), shape_a);
    auto shape_r = Shape{3, 4};
    auto rt = make_shared<TensorViewType>(element::Int64::element_type(), shape_r);
    auto f = make_shared<Function>(
        make_shared<op::Broadcast>(A, shape_r, AxisSet{0}), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Int64>(shape_a);
    *a = vector<element::Int64::type>{1, 2, 3, 4};
    auto result = backend->make_parameterized_tensor_view<element::Int64>(shape_r);

    (*cf)({a}, {result});
    ASSERT_EQ((vector<element::Int64::type>{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}),
              result->get_vector());
}

TEST(execute, convert_int32_float32)
{
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Int32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(
        make_shared<op::Convert>(A, element::Float32::element_type()), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Int32>(shape);
    *a = vector<element::Int32::type>{1, 2, 3, 4};
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape);

    (*cf)({a}, {result});
    ASSERT_EQ((vector<element::Float32::type>{1, 2, 3, 4}), result->get_vector());
}

TEST(execute, convert_int32_bool)
{
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Int32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Bool::element_type(), shape);
    auto f = make_shared<Function>(
        make_shared<op::Convert>(A, element::Bool::element_type()), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Int32>(shape);
    *a = vector<element::Int32::type>{1, 2, 3, 4};
    auto result = backend->make_parameterized_tensor_view<element::Bool>(shape);

    (*cf)({a}, {result});
    ASSERT_EQ((vector<element::Bool::type>{1, 2, 3, 4}), result->get_vector());
}

TEST(execute, convert_float32_bool)
{
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Bool::element_type(), shape);
    auto f = make_shared<Function>(
        make_shared<op::Convert>(A, element::Bool::element_type()), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *a = vector<element::Float32::type>{1, 2, 3, 4};
    auto result = backend->make_parameterized_tensor_view<element::Bool>(shape);

    (*cf)({a}, {result});
    ASSERT_EQ((vector<element::Bool::type>{1, 2, 3, 4}), result->get_vector());
}

// Trivial case with no reduction axes.
TEST(execute, reduce_trivial)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
    auto f_B = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
    auto f_rt = make_shared<TensorViewType>(element::Float32::element_type(), Shape{});
    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), f_rt, op::Parameters{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    auto shape = Shape{2, 2};
    auto g_A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto g_B = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
    auto g_rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto g = make_shared<Function>(
        make_shared<op::Reduce>(g_A, g_B, f, AxisSet{}), g_rt, op::Parameters{g_A, g_B});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(g);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = ngraph::runtime::make_tensor<element::Float32>(shape);
    *a = vector<float>{1, 2, 3, 4};
    auto b = ngraph::runtime::make_tensor<element::Float32>(shape);
    *b = vector<float>{0};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<float>{1, 2, 3, 4}), result->get_vector());
}

TEST(execute, reduce_to_scalar)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
    auto f_B = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
    auto f_rt = make_shared<TensorViewType>(element::Float32::element_type(), Shape{});
    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), f_rt, op::Parameters{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    auto shape = Shape{2, 2};
    auto g_A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto g_B = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
    auto g_rt = make_shared<TensorViewType>(element::Float32::element_type(), Shape{});
    auto g = make_shared<Function>(
        make_shared<op::Reduce>(g_A, g_B, f, AxisSet{0, 1}), g_rt, op::Parameters{g_A, g_B});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(g);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = ngraph::runtime::make_tensor<element::Float32>(shape);
    *a = vector<float>{1, 2, 3, 4};
    auto b = ngraph::runtime::make_tensor<element::Float32>(Shape{});
    *b = vector<float>{0};
    auto result = ngraph::runtime::make_tensor<element::Float32>(Shape{});

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<float>{10}), result->get_vector());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    ASSERT_EQ((vector<float>{1, 2, 3, 4}), a->get_vector());
    ASSERT_EQ((vector<float>{0}), b->get_vector());
}

TEST(execute, reduce_matrix_columns)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
    auto f_B = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
    auto f_rt = make_shared<TensorViewType>(element::Float32::element_type(), Shape{});
    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), f_rt, op::Parameters{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    auto shape_a = Shape{3, 2};
    auto g_A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto g_B = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
    auto shape_rt = Shape{2};
    auto g_rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_rt);
    auto g = make_shared<Function>(
        make_shared<op::Reduce>(g_A, g_B, f, AxisSet{0}), g_rt, op::Parameters{g_A, g_B});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(g);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = ngraph::runtime::make_tensor<element::Float32>(shape_a);
    *a = vector<float>{1, 2, 3, 4, 5, 6};
    auto b = ngraph::runtime::make_tensor<element::Float32>(Shape{});
    *b = vector<float>{0};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape_rt);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<float>{9, 12}), result->get_vector());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    ASSERT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), a->get_vector());
    ASSERT_EQ((vector<float>{0}), b->get_vector());
}

TEST(execute, reduce_matrix_rows)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
    auto f_B = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
    auto f_rt = make_shared<TensorViewType>(element::Float32::element_type(), Shape{});
    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), f_rt, op::Parameters{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    auto shape_a = Shape{3, 2};
    auto g_A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto g_B = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
    auto shape_rt = Shape{3};
    auto g_rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_rt);
    auto g = make_shared<Function>(
        make_shared<op::Reduce>(g_A, g_B, f, AxisSet{1}), g_rt, op::Parameters{g_A, g_B});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(g);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = ngraph::runtime::make_tensor<element::Float32>(shape_a);
    *a = vector<float>{1, 2, 3, 4, 5, 6};
    auto b = ngraph::runtime::make_tensor<element::Float32>(Shape{});
    *b = vector<float>{0};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape_rt);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<float>{3, 7, 11}), result->get_vector());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    ASSERT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), a->get_vector());
    ASSERT_EQ((vector<float>{0}), b->get_vector());
}

TEST(execute, reduce_matrix_rows_zero)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
    auto f_B = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
    auto f_rt = make_shared<TensorViewType>(element::Float32::element_type(), Shape{});
    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), f_rt, op::Parameters{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    auto shape_a = Shape{3, 0};
    auto g_A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto g_B = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
    auto shape_rt = Shape{3};
    auto g_rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_rt);
    auto g = make_shared<Function>(
        make_shared<op::Reduce>(g_A, g_B, f, AxisSet{1}), g_rt, op::Parameters{g_A, g_B});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(g);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = ngraph::runtime::make_tensor<element::Float32>(shape_a);
    *a = vector<float>{};
    auto b = ngraph::runtime::make_tensor<element::Float32>(Shape{});
    *b = vector<float>{66};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape_rt);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<float>{66, 66, 66}), result->get_vector());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    ASSERT_EQ((vector<float>{}), a->get_vector());
    ASSERT_EQ((vector<float>{66}), b->get_vector());
}

TEST(execute, reduce_matrix_cols_zero)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
    auto f_B = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
    auto f_rt = make_shared<TensorViewType>(element::Float32::element_type(), Shape{});
    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), f_rt, op::Parameters{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    auto shape_a = Shape{0, 2};
    auto g_A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto g_B = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
    auto shape_rt = Shape{2};
    auto g_rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_rt);
    auto g = make_shared<Function>(
        make_shared<op::Reduce>(g_A, g_B, f, AxisSet{0}), g_rt, op::Parameters{g_A, g_B});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(g);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = ngraph::runtime::make_tensor<element::Float32>(shape_a);
    *a = vector<float>{};
    auto b = ngraph::runtime::make_tensor<element::Float32>(Shape{});
    *b = vector<float>{77};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape_rt);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<float>{77, 77}), result->get_vector());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    ASSERT_EQ((vector<float>{}), a->get_vector());
    ASSERT_EQ((vector<float>{77}), b->get_vector());
}

TEST(execute, reduce_vector_zero)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
    auto f_B = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
    auto f_rt = make_shared<TensorViewType>(element::Float32::element_type(), Shape{});
    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), f_rt, op::Parameters{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    auto shape_a = Shape{0};
    auto g_A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto g_B = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
    auto shape_rt = Shape{};
    auto g_rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_rt);
    auto g = make_shared<Function>(
        make_shared<op::Reduce>(g_A, g_B, f, AxisSet{0}), g_rt, op::Parameters{g_A, g_B});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(g);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = ngraph::runtime::make_tensor<element::Float32>(shape_a);
    *a = vector<float>{};
    auto b = ngraph::runtime::make_tensor<element::Float32>(Shape{});
    *b = vector<float>{88};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape_rt);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<float>{88}), result->get_vector());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    ASSERT_EQ((vector<float>{}), a->get_vector());
    ASSERT_EQ((vector<float>{88}), b->get_vector());
}

TEST(execute, reduce_matrix_to_scalar_zero_by_zero)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
    auto f_B = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
    auto f_rt = make_shared<TensorViewType>(element::Float32::element_type(), Shape{});
    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), f_rt, op::Parameters{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    auto shape_a = Shape{0, 0};
    auto g_A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto g_B = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
    auto shape_rt = Shape{};
    auto g_rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_rt);
    auto g = make_shared<Function>(
        make_shared<op::Reduce>(g_A, g_B, f, AxisSet{0, 1}), g_rt, op::Parameters{g_A, g_B});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(g);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = ngraph::runtime::make_tensor<element::Float32>(shape_a);
    *a = vector<float>{};
    auto b = ngraph::runtime::make_tensor<element::Float32>(Shape{});
    *b = vector<float>{99};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape_rt);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<float>{99}), result->get_vector());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    ASSERT_EQ((vector<float>{}), a->get_vector());
    ASSERT_EQ((vector<float>{99}), b->get_vector());
}

TEST(execute, reshape_t2v_012)
{
    auto shape_a = Shape{2, 2, 3};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_r = Shape{12};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Reshape>(A, AxisVector{0, 1, 2}, shape_r);
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = ngraph::runtime::make_tensor<element::Float32>(shape_a);
    *a = vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape_r);

    (*cf)({a}, {result});
    ASSERT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}), result->get_vector());
}

TEST(execute, reshape_t2s_012)
{
    auto shape_a = Shape{1, 1, 1};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_r = Shape{};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Reshape>(A, AxisVector{0, 1, 2}, shape_r);
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = ngraph::runtime::make_tensor<element::Float32>(shape_a);
    *a = vector<float>{6};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape_r);

    (*cf)({a}, {result});
    ASSERT_EQ((vector<float>{6}), result->get_vector());
}

TEST(execute, reshape_t2s_120)
{
    auto shape_a = Shape{1, 1, 1};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_r = Shape{};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Reshape>(A, AxisVector{1, 2, 0}, shape_r);
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = ngraph::runtime::make_tensor<element::Float32>(shape_a);
    *a = vector<float>{6};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape_r);

    (*cf)({a}, {result});
    ASSERT_EQ((vector<float>{6}), result->get_vector());
}

TEST(execute, reshape_s2t)
{
    auto shape_a = Shape{};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_r = Shape{1, 1, 1, 1, 1, 1};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Reshape>(A, AxisVector{}, shape_r);
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = ngraph::runtime::make_tensor<element::Float32>(shape_a);
    *a = vector<float>{42};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape_r);

    (*cf)({a}, {result});
    ASSERT_EQ((vector<float>{42}), result->get_vector());
}

TEST(execute, reshape_v2m_col)
{
    auto shape_a = Shape{3};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_r = Shape{3, 1};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Reshape>(A, AxisVector{0}, shape_r);
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = ngraph::runtime::make_tensor<element::Float32>(shape_a);
    *a = vector<float>{1, 2, 3};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape_r);

    (*cf)({a}, {result});
    ASSERT_EQ((vector<float>{1, 2, 3}), result->get_vector());
}

TEST(execute, reshape_v2m_row)
{
    auto shape_a = Shape{3};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_r = Shape{1, 3};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Reshape>(A, AxisVector{0}, shape_r);
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = ngraph::runtime::make_tensor<element::Float32>(shape_a);
    *a = vector<float>{1, 2, 3};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape_r);

    (*cf)({a}, {result});
    ASSERT_EQ((vector<float>{1, 2, 3}), result->get_vector());
}

TEST(execute, reshape_v2t_middle)
{
    auto shape_a = Shape{3};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_r = Shape{1, 3, 1};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Reshape>(A, AxisVector{0}, shape_r);
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = ngraph::runtime::make_tensor<element::Float32>(shape_a);
    *a = vector<float>{1, 2, 3};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape_r);

    (*cf)({a}, {result});
    ASSERT_EQ((vector<float>{1, 2, 3}), result->get_vector());
}

TEST(execute, reshape_m2m_same)
{
    auto shape_a = Shape{3, 3};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_r = Shape{3, 3};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Reshape>(A, AxisVector{0, 1}, shape_r);
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = ngraph::runtime::make_tensor<element::Float32>(shape_a);
    *a = vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape_r);

    (*cf)({a}, {result});
    ASSERT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9}), result->get_vector());
}

TEST(execute, reshape_m2m_transpose)
{
    auto shape_a = Shape{3, 3};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_r = Shape{3, 3};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Reshape>(A, AxisVector{1, 0}, shape_r);
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = ngraph::runtime::make_tensor<element::Float32>(shape_a);
    *a = vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape_r);

    (*cf)({a}, {result});
    ASSERT_EQ((vector<float>{1, 4, 7, 2, 5, 8, 3, 6, 9}), result->get_vector());
}

TEST(execute, reshape_m2m_dim_change_transpose)
{
    auto shape_a = Shape{3, 2};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_r = Shape{2, 3};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Reshape>(A, AxisVector{1, 0}, shape_r);
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = ngraph::runtime::make_tensor<element::Float32>(shape_a);
    *a = vector<float>{1, 2, 3, 4, 5, 6};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape_r);

    (*cf)({a}, {result});
    ASSERT_EQ((vector<float>{1, 3, 5, 2, 4, 6}), result->get_vector());
}

TEST(execute, sin)
{
    auto shape = Shape{6};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto result_type = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Sin>(A), result_type, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    float pi = acosf(-1);
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape);
    vector<float> input{pi / 2, 0.0f, -0.0f, pi / 6, -pi, pi};
    *a = input;
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float f) -> float { return sinf(f); });

    (*cf)({a}, {result});
    ASSERT_EQ(input, result->get_vector());
}

TEST(execute, cos)
{
    auto shape = Shape{6};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto result_type = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Cos>(A), result_type, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    float pi = acosf(-1);
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape);
    vector<float> input{pi / 2, 0.0f, -0.0f, pi / 3, -pi, pi};
    *a = input;
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float f) -> float { return cosf(f); });

    (*cf)({a}, {result});
    ASSERT_EQ(input, result->get_vector());
}

TEST(execute, tan)
{
    auto shape = Shape{6};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto result_type = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Tan>(A), result_type, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    float pi = acosf(-1);
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape);
    vector<float> input{pi / 4, 0.0f, -0.0f, 7 * pi / 4, 3 * pi / 4, 5 * pi / 4};
    *a = input;
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float f) -> float { return tanf(f); });

    (*cf)({a}, {result});
    ASSERT_EQ(input, result->get_vector());
}

TEST(execute, asin)
{
    auto shape = Shape{6};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto result_type = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Asin>(A), result_type, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape);
    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 0.5f, -0.5f};
    *a = input;
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float f) -> float { return asinf(f); });

    (*cf)({a}, {result});
    ASSERT_EQ(input, result->get_vector());
}

TEST(execute, acos)
{
    auto shape = Shape{6};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto result_type = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Acos>(A), result_type, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape);
    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 0.5f, -0.5f};
    *a = input;
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float f) -> float { return acosf(f); });

    (*cf)({a}, {result});
    ASSERT_EQ(input, result->get_vector());
}

TEST(execute, atan)
{
    auto shape = Shape{6};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto result_type = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Atan>(A), result_type, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape);
    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 0.5f, -0.5f};
    *a = input;
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float f) -> float { return atanf(f); });

    (*cf)({a}, {result});
    ASSERT_EQ(input, result->get_vector());
}

TEST(execute, sinh)
{
    auto shape = Shape{6};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto result_type = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Sinh>(A), result_type, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape);
    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 5.0f, -5.0f};
    *a = input;
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float f) -> float { return sinhf(f); });

    (*cf)({a}, {result});
    ASSERT_EQ(input, result->get_vector());
}

TEST(execute, cosh)
{
    auto shape = Shape{6};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto result_type = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Cosh>(A), result_type, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape);
    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 5.0f, -5.0f};
    *a = input;
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float f) -> float { return coshf(f); });

    (*cf)({a}, {result});
    ASSERT_EQ(input, result->get_vector());
}

TEST(execute, tanh)
{
    auto shape = Shape{6};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto result_type = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Tanh>(A), result_type, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape);
    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 0.5f, -0.5f};
    *a = input;
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float f) -> float { return tanhf(f); });

    (*cf)({a}, {result});
    ASSERT_EQ(input, result->get_vector());
}

TEST(execute, exp)
{
    auto shape = Shape{8};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto result_type = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Exp>(A), result_type, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *a = vector<float>{-4, -3, -2, -1, 0, 1, 2, 3};
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape);

    (*cf)({a}, {result});
    ASSERT_EQ(
        (vector<float>{expf(-4), expf(-3), expf(-2), expf(-1), expf(0), expf(1), expf(2), expf(3)}),
        result->get_vector());
}

TEST(execute, slice_scalar)
{
    auto shape_a = Shape{};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_r = Shape{};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Slice>(A, Coordinate{}, Coordinate{});
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = ngraph::runtime::make_tensor<element::Float32>(shape_a);
    *a = vector<float>{312};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape_r);

    (*cf)({a}, {result});
    ASSERT_EQ((vector<float>{312}), result->get_vector());
}

TEST(execute, slice_matrix)
{
    auto shape_a = Shape{4, 4};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_r = Shape{2, 3};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Slice>(A, Coordinate{0, 1}, Coordinate{3, 3});
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = ngraph::runtime::make_tensor<element::Float32>(shape_a);
    *a = vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape_r);

    (*cf)({a}, {result});
    ASSERT_EQ((vector<float>{2, 3, 6, 7, 10, 11}), result->get_vector());
}

TEST(execute, slice_vector)
{
    auto shape_a = Shape{16};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_r = Shape{12};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Slice>(A, Coordinate{2}, Coordinate{14});
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = ngraph::runtime::make_tensor<element::Float32>(shape_a);
    *a = vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape_r);

    (*cf)({a}, {result});
    ASSERT_EQ((vector<float>{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}), result->get_vector());
}
