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
#include <cinttypes>
#include <cmath>

#include "gtest/gtest.h"

#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;

template <typename T>
static void copy_data(shared_ptr<runtime::TensorView> tv, const vector<T>& data)
{
    size_t data_size = data.size() * sizeof(T);
    tv->write(data.data(), 0, data_size);
}

TEST(${BACKEND_NAME}, abc)
{
    using f32 = element::Float32;

    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(f32::element_type(), shape);
    auto B = make_shared<op::Parameter>(f32::element_type(), shape);
    auto C = make_shared<op::Parameter>(f32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(f32::element_type(), shape);
    auto f = make_shared<Function>((A + B) * C, rt, op::Parameters{A, B, C});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    shared_ptr<runtime::TensorView> a =
        backend->make_primary_tensor_view(f32::element_type(), shape);
    shared_ptr<runtime::TensorView> b =
        backend->make_primary_tensor_view(f32::element_type(), shape);
    shared_ptr<runtime::TensorView> c =
        backend->make_primary_tensor_view(f32::element_type(), shape);
    shared_ptr<runtime::TensorView> result =
        backend->make_primary_tensor_view(f32::element_type(), shape);

    copy_data(a, runtime::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, runtime::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, runtime::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    cf->call({a, b, c}, {result});
    EXPECT_EQ(*result, (runtime::NDArray<float, 2>({{54, 80}, {110, 144}})));

    cf->call({b, a, c}, {result});
    EXPECT_EQ(*result, (runtime::NDArray<float, 2>({{54, 80}, {110, 144}})));

    cf->call({a, c, b}, {result});
    EXPECT_EQ(*result, (runtime::NDArray<float, 2>({{50, 72}, {98, 128}})));
}

TEST(${BACKEND_NAME}, abc_int64)
{
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Int64::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Int64::element_type(), shape);
    auto C = make_shared<op::Parameter>(element::Int64::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Int64::element_type(), shape);
    auto f = make_shared<Function>((A + B) * C, rt, op::Parameters{A, B, C});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Int64::element_type(), shape);
    copy_data(a, vector<element::Int64::type>{1, 2, 3, 4});
    auto b = backend->make_primary_tensor_view(element::Int64::element_type(), shape);
    copy_data(b, vector<element::Int64::type>{5, 6, 7, 8});
    auto c = backend->make_primary_tensor_view(element::Int64::element_type(), shape);
    copy_data(c, vector<element::Int64::type>{9, 10, 11, 12});
    auto result = backend->make_primary_tensor_view(element::Int64::element_type(), shape);

    cf->call({a, b, c}, {result});
    EXPECT_EQ((vector<element::Int64::type>{54, 80, 110, 144}), result->get_vector<int64_t>());

    cf->call({b, a, c}, {result});
    EXPECT_EQ((vector<element::Int64::type>{54, 80, 110, 144}), result->get_vector<int64_t>());

    cf->call({a, c, b}, {result});
    EXPECT_EQ((vector<element::Int64::type>{50, 72, 98, 128}), result->get_vector<int64_t>());
}

// Same as abc, but using tuples for input and output
TEST(${BACKEND_NAME}, abc_tuple)
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

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(b, vector<float>{5, 6, 7, 8});
    auto c = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(c, vector<float>{9, 10, 11, 12});
    auto abc = runtime::make_tuple({a, b, c});
    auto bac = runtime::make_tuple({b, a, c});
    auto acb = runtime::make_tuple({a, c, b});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    auto result_tuple = runtime::make_tuple({result});

    cf->call({abc}, {result_tuple});
    ASSERT_EQ((vector<float>{54, 80, 110, 144}), result->get_vector<float>());

    cf->call({bac}, {result_tuple});
    ASSERT_EQ((vector<float>{54, 80, 110, 144}), result->get_vector<float>());

    cf->call({acb}, {result_tuple});
    ASSERT_EQ((vector<float>{50, 72, 98, 128}), result->get_vector<float>());
}

// Same as abc, but using tuples for input and output
TEST(${BACKEND_NAME}, abc_tuple_int64)
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

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Int64::element_type(), shape);
    copy_data(a, vector<element::Int64::type>{1, 2, 3, 4});
    auto b = backend->make_primary_tensor_view(element::Int64::element_type(), shape);
    copy_data(b, vector<element::Int64::type>{5, 6, 7, 8});
    auto c = backend->make_primary_tensor_view(element::Int64::element_type(), shape);
    copy_data(c, vector<element::Int64::type>{9, 10, 11, 12});
    auto abc = runtime::make_tuple({a, b, c});
    auto bac = runtime::make_tuple({b, a, c});
    auto acb = runtime::make_tuple({a, c, b});
    auto result = backend->make_primary_tensor_view(element::Int64::element_type(), shape);
    auto result_tuple = runtime::make_tuple({result});

    cf->call({abc}, {result_tuple});
    ASSERT_EQ((vector<element::Int64::type>{54, 80, 110, 144}),
              result->get_vector<element::Int64::type>());

    cf->call({bac}, {result_tuple});
    ASSERT_EQ((vector<element::Int64::type>{54, 80, 110, 144}),
              result->get_vector<element::Int64::type>());

    cf->call({acb}, {result_tuple});
    ASSERT_EQ((vector<element::Int64::type>{50, 72, 98, 128}),
              result->get_vector<element::Int64::type>());
}

// Multiple retrive values
TEST(${BACKEND_NAME}, tuple_result)
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

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(b, vector<float>{5, 6, 7, 8});
    auto c = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(c, vector<float>{9, 10, 11, 12});

    auto r0 = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    auto r1 = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    auto result_tuple = runtime::make_tuple({r0, r1});

    cf->call({a, b, c}, {result_tuple});

    ASSERT_EQ((vector<float>{6, 8, 10, 12}), r0->get_vector<float>());
    ASSERT_EQ((vector<float>{54, 80, 110, 144}), r1->get_vector<float>());
}

TEST(${BACKEND_NAME}, abs)
{
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto result_type = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Abs>(A), result_type, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{1, -2, 0, -4.8f});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{1, 2, 0, 4.8f}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, ceiling)
{
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto result_type = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Ceiling>(A), result_type, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{-2.5f, -2.0f, 0.3f, 4.8f});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{-2.0f, -2.0f, 1.0f, 5.0f}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, concat_matrix_colwise)
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

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape_b);
    copy_data(b, vector<float>{1, 2, 4, 8, 16, 32});
    auto c = backend->make_primary_tensor_view(element::Float32::element_type(), shape_c);
    copy_data(c, vector<float>{2, 3, 5, 7, 11, 13});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a, b, c}, {result});
    ASSERT_EQ((vector<float>{2, 4, 1, 2, 4, 2, 3, 5, 8, 16, 8, 16, 32, 7, 11, 13}),
              result->get_vector<float>());
}

TEST(${BACKEND_NAME}, concat_matrix_rowwise)
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

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape_b);
    copy_data(b, vector<float>{1, 2, 4, 8, 16, 32});
    auto c = backend->make_primary_tensor_view(element::Float32::element_type(), shape_c);
    copy_data(c, vector<float>{2, 3, 5, 7, 11, 13});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a, b, c}, {result});
    ASSERT_EQ((vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13}),
              result->get_vector<float>());
}

TEST(${BACKEND_NAME}, concat_matrix_int64)
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

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Int64::element_type(), shape_a);
    copy_data(a, vector<element::Int64::type>{2, 4, 8, 16});
    auto b = backend->make_primary_tensor_view(element::Int64::element_type(), shape_b);
    copy_data(b, vector<element::Int64::type>{1, 2, 4, 8, 16, 32});
    auto c = backend->make_primary_tensor_view(element::Int64::element_type(), shape_c);
    copy_data(c, vector<element::Int64::type>{2, 3, 5, 7, 11, 13});
    auto result = backend->make_primary_tensor_view(element::Int64::element_type(), shape_r);

    cf->call({a, b, c}, {result});
    ASSERT_EQ((vector<element::Int64::type>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13}),
              result->get_vector<element::Int64::type>());
}

TEST(${BACKEND_NAME}, concat_vector)
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

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape_b);
    copy_data(b, vector<float>{1, 2, 4, 8, 16, 32});
    auto c = backend->make_primary_tensor_view(element::Float32::element_type(), shape_c);
    copy_data(c, vector<float>{18, 19});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a, b, c}, {result});
    ASSERT_EQ((vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 18, 19}),
              result->get_vector<float>());
}

TEST(${BACKEND_NAME}, divide)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
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
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(b, vector<float>{1, 2, 4, 8});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<float>{2, 2, 2, 2}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, divide_by_zero_float32)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
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
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(b, vector<float>{0, 0, 0, 0});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<float>{std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity(),
                             std::numeric_limits<float>::infinity()}),
              result->get_vector<float>());
}

TEST(${BACKEND_NAME}, divide_by_zero_int32)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    auto shape = Shape{2, 2};

    auto make_external = [&]() {
        auto A = make_shared<op::Parameter>(element::Int32::element_type(), shape);
        auto B = make_shared<op::Parameter>(element::Int32::element_type(), shape);
        auto rt = make_shared<TensorViewType>(element::Int32::element_type(), shape);
        auto f = make_shared<Function>(make_shared<op::Divide>(A, B), rt, op::Parameters{A, B});

        auto external = manager->compile(f);
        return external;
    };

    auto cf = backend->make_call_frame(make_external());

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Int32::element_type(), shape);
    copy_data(a, vector<int>{2, 4, 8, 16});
    auto b = backend->make_primary_tensor_view(element::Int32::element_type(), shape);
    copy_data(b, vector<int>{0, 0, 0, 0});
    auto result = backend->make_primary_tensor_view(element::Int32::element_type(), shape);

    EXPECT_ANY_THROW({ cf->call({a, b}, {result}); });
}

TEST(${BACKEND_NAME}, equal)
{
    auto shape = Shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Bool::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Equal>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0, 1, 1});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(b, vector<float>{1, 8, 4, 8, 0, 0, 1, 1.5});
    auto result = backend->make_primary_tensor_view(element::Bool::element_type(), shape);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<char>{1, 1, 0, 0, 0, 1, 1, 0}), result->get_vector<char>());
}

TEST(${BACKEND_NAME}, floor)
{
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto result_type = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Floor>(A), result_type, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{-2.5f, -2.0f, 0.3f, 4.8f});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{-3.0f, -2.0f, 0.0f, 4.0f}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, dot_0_0)
{
    auto shape = Shape{0};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto shape_r = Shape{};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), Shape{});
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(b, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the right value.
    copy_data(result, vector<float>{2112});

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<float>{0}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, dot_matrix_2x0_0x2)
{
    auto shape_a = Shape{2, 0};
    auto shape_b = Shape{0, 2};
    auto shape_r = Shape{2, 2};

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
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
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape_b);
    copy_data(b, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the right value.
    copy_data(result, vector<float>{2112, 2112, 2112, 2112});

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<float>{0, 0, 0, 0}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, dot_matrix_0x2_2x0)
{
    auto shape_a = Shape{0, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_b = Shape{2, 0};
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape_b);
    auto shape_r = Shape{0, 0};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape_b);
    copy_data(b, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<float>{}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, dot_matrix_3x2_2x0)
{
    auto shape_a = Shape{3, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_b = Shape{2, 0};
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape_b);
    auto shape_r = Shape{0, 0};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape_b);
    copy_data(b, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<float>{}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, dot_scalar_0x2)
{
    auto shape_a = Shape{};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_b = Shape{0, 2};
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape_b);
    auto shape_r = Shape{0, 2};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{1});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape_b);
    copy_data(b, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<float>{}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, dot_2x0_0)
{
    auto shape_a = Shape{2, 0};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_b = Shape{0};
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape_b);
    auto shape_r = Shape{2};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape_b);
    copy_data(b, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the right value.
    copy_data(result, vector<float>{2112, 2112});

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<float>{0, 0}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, dot1d)
{
    auto shape = Shape{4};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto shape_r = Shape{1};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), Shape{});
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(b, vector<float>{1, 2, 4, 8});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<float>{170}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, dot2d)
{
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto shape_r = Shape{2, 2};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(b, vector<float>{5, 6, 7, 8});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<float>{19, 22, 43, 50}), result->get_vector<float>());
}

//
// Here is what numpy does:
//
// >>> a = linspace(1,2*2*2,2*2*2)
// >>> b = linspace(1,2*2*2,2*2*2)
//
// >>> a.shape=(2,2,2)
// >>> b.shape=(2,2,2)
//
// >>> tensordot(a,b,axes=([2],[1]))
// array([[[[   7.,   10.],
//          [  19.,   22.]],
//
//         [[  15.,   22.],
//          [  43.,   50.]]],
//
//
//        [[[  23.,   34.],
//          [  67.,   78.]],
//
//         [[  31.,   46.],
//          [  91.,  106.]]]])
//
TEST(${BACKEND_NAME}, dot3d_3d)
{
    auto shape = Shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto shape_r = Shape{2, 2, 2, 2};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(b, vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<float>{7, 10, 19, 22, 15, 22, 43, 50, 23, 34, 67, 78, 31, 46, 91, 106}),
              result->get_vector<float>());
}

//
// Here is what numpy does:
//
// >>> from numpy import *
// >>> a = linspace(0,4*2*3-1,4*2*3)
// >>> b = linspace(0,3*4-1,3*4)
//
// >>> a.shape=(4,2,3)
// >>> b.shape=(3,4)
//
// >>> tensordot(a,b,axes=([2],[0]))
// array([[[  20.,   23.,   26.,   29.],
//         [  56.,   68.,   80.,   92.]],
//
//        [[  92.,  113.,  134.,  155.],
//         [ 128.,  158.,  188.,  218.]],
//
//        [[ 164.,  203.,  242.,  281.],
//         [ 200.,  248.,  296.,  344.]],
//
//        [[ 236.,  293.,  350.,  407.],
//         [ 272.,  338.,  404.,  470.]]])
//
TEST(${BACKEND_NAME}, dot3d_2d)
{
    auto shape_a = Shape{4, 2, 3};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_b = Shape{3, 4};
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape_b);
    auto shape_r = Shape{4, 2, 4};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                               12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape_b);
    copy_data(b, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<float>{20,  23,  26,  29,  56,  68,  80,  92,  92,  113, 134,
                             155, 128, 158, 188, 218, 164, 203, 242, 281, 200, 248,
                             296, 344, 236, 293, 350, 407, 272, 338, 404, 470}),
              result->get_vector<float>());
}

TEST(${BACKEND_NAME}, dot_scalar_tensor_arg0)
{
    auto shape_a = Shape{};
    auto shape_b = Shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape_b);
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_b);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{6});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape_b);
    copy_data(b, vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_b);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<float>{6, 12, 18, 24, 30, 36, 42, 48}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, dot_scalar_tensor_arg1)
{
    auto shape_a = Shape{2, 2, 2};
    auto shape_b = Shape{};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape_b);
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_a);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape_b);
    copy_data(b, vector<float>{6});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<float>{6, 12, 18, 24, 30, 36, 42, 48}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, dot_scalar_scalar)
{
    auto shape = Shape{};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{8});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(b, vector<float>{6});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<float>{48}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, dot_matrix_vector)
{
    auto shape_a = Shape{4, 4};
    auto shape_b = Shape{4};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape_b);
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_b);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), rt, op::Parameters{A, B});
    auto shape_r = Shape{4};

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape_b);
    copy_data(b, vector<float>{17, 18, 19, 20});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<float>{190, 486, 782, 1078}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, dot_matrix_vector_int64)
{
    auto shape_a = Shape{4, 4};
    auto shape_b = Shape{4};
    auto A = make_shared<op::Parameter>(element::Int64::element_type(), shape_a);
    auto B = make_shared<op::Parameter>(element::Int64::element_type(), shape_b);
    auto rt = make_shared<TensorViewType>(element::Int64::element_type(), shape_b);
    auto f = make_shared<Function>(make_shared<op::Dot>(A, B), rt, op::Parameters{A, B});
    auto shape_r = Shape{4};

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Int64::element_type(), shape_a);
    copy_data(a,
              vector<element::Int64::type>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto b = backend->make_primary_tensor_view(element::Int64::element_type(), shape_b);
    copy_data(b, vector<element::Int64::type>{17, 18, 19, 20});
    auto result = backend->make_primary_tensor_view(element::Int64::element_type(), shape_r);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<element::Int64::type>{190, 486, 782, 1078}),
              result->get_vector<element::Int64::type>());
}

TEST(${BACKEND_NAME}, greater)
{
    auto shape = Shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Bool::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Greater>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0.5, 2, 1});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(b, vector<float>{1, 2, 4, 8, 0, 0, 1, 1.5});
    auto result = backend->make_primary_tensor_view(element::Bool::element_type(), shape);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<char>{0, 1, 0, 1, 0, 1, 1, 0}), result->get_vector<char>());
}

TEST(${BACKEND_NAME}, greatereq)
{
    auto shape = Shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Bool::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::GreaterEq>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0, 2, 1});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(b, vector<float>{1, 2, -8, 8, 0, 0, 0.5, 1.5});
    auto result = backend->make_primary_tensor_view(element::Bool::element_type(), shape);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<char>{1, 1, 1, 1, 0, 1, 1, 0}), result->get_vector<char>());
}

TEST(${BACKEND_NAME}, less)
{
    auto shape = Shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Bool::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Less>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0.5, 2, 1});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(b, vector<float>{1, 2, 4, 8, 0, 0, 1, 1.5});
    auto result = backend->make_primary_tensor_view(element::Bool::element_type(), shape);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<char>{0, 0, 1, 0, 1, 0, 0, 1}), result->get_vector<char>());
}

TEST(${BACKEND_NAME}, lesseq)
{
    auto shape = Shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Bool::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::LessEq>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0, 2, 1});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(b, vector<float>{1, 2, -8, 8, 0, 0, 0.5, 1.5});
    auto result = backend->make_primary_tensor_view(element::Bool::element_type(), shape);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<char>{1, 0, 1, 0, 1, 1, 0, 1}), result->get_vector<char>());
}

TEST(${BACKEND_NAME}, lesseq_bool)
{
    auto shape = Shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Bool::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Bool::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Bool::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::LessEq>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Bool::element_type(), shape);
    copy_data(a, vector<char>{1, 1, 1, 1, 1, 1, 1, 1});
    auto b = backend->make_primary_tensor_view(element::Bool::element_type(), shape);
    copy_data(b, vector<char>{0, 0, 0, 0, 0, 0, 0, 0});
    auto result = backend->make_primary_tensor_view(element::Bool::element_type(), shape);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the right value.
    copy_data(result, vector<char>{1, 1, 1, 1, 1, 1, 1, 1});

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<char>{0, 0, 0, 0, 0, 0, 0, 0}), result->get_vector<char>());
}

TEST(${BACKEND_NAME}, log)
{
    auto shape = Shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Log>(A), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(
        a, vector<float>{expf(1), expf(2), expf(3), expf(4), expf(5), expf(6), expf(7), expf(8)});
    vector<float> loga;
    for (auto elt : a->get_vector<float>())
    {
        loga.push_back(logf(elt));
    }
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    cf->call({a}, {result});
    ASSERT_EQ(loga, result->get_vector<float>());
}

TEST(${BACKEND_NAME}, maximum)
{
    auto shape = Shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Maximum>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0.5, 2, 1});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(b, vector<float>{1, 2, 4, 8, 0, 0, 1, 1.5});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<float>{1, 8, 4, 17, 0, 0.5, 2, 1.5}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, minimum)
{
    auto shape = Shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Minimum>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0.5, 2, 1});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(b, vector<float>{1, 2, 4, 8, 0, 0, 1, 1.5});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<float>{1, 2, -8, 8, -.5, 0, 1, 1}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, negative)
{
    auto shape = Shape{2, 3};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Negative>(A), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{1, -2, 0, -4.8f, 8.6f, -8.6f});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{-1, 2, 0, 4.8f, -8.6f, 8.6f}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, notequal)
{
    auto shape = Shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Bool::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::NotEqual>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{1, 8, -8, 17, -0.5, 0, 1, 1});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(b, vector<float>{1, 8, 4, 8, 0, 0, 1, 1.5});
    auto result = backend->make_primary_tensor_view(element::Bool::element_type(), shape);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<char>{0, 0, 1, 1, 1, 0, 0, 1}), result->get_vector<char>());
}

TEST(${BACKEND_NAME}, select)
{
    auto shape = Shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Bool::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto C = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Select>(A, B, C), rt, op::Parameters{A, B, C});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Bool::element_type(), shape);
    copy_data(a, vector<char>{0, 1, 1, 0, 0, 1, 0, 1});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(b, vector<float>{1, 2, 3, 4, 5, 6, 7, 8});
    auto c = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(c, vector<float>{11, 12, 13, 14, 15, 16, 17, 18});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    cf->call({a, b, c}, {result});
    ASSERT_EQ((vector<float>{11, 2, 3, 14, 15, 6, 17, 8}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, subtract)
{
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Subtract>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{2, 4, 8, 16});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(b, vector<float>{1, 2, 4, 8});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<float>{1, 2, 4, 8}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, scalar_parameterized_constant_bool)
{
    auto shape = Shape{};
    auto t = runtime::make_tensor<element::Bool>(shape, {true});
    auto A = make_shared<op::ParameterizedConstant<element::Bool>>(shape, t);
    auto rt = make_shared<TensorViewType>(element::Bool::element_type(), shape);
    auto f = make_shared<Function>(A, rt, op::Parameters{});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_primary_tensor_view(element::Bool::element_type(), shape);

    cf->call({}, {result});
    ASSERT_EQ((vector<char>{true}), result->get_vector<char>());
}

TEST(${BACKEND_NAME}, scalar_parameterized_constant_float)
{
    auto shape = Shape{};
    auto t = runtime::make_tensor<element::Float32>(shape, {-3.0f});
    auto A = make_shared<op::ParameterizedConstant<element::Float32>>(shape, t);
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(A, rt, op::Parameters{});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    cf->call({}, {result});
    ASSERT_EQ((vector<float>{-3.0f}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, scalar_parameterized_constant_int8)
{
    auto shape = Shape{};
    auto t = runtime::make_tensor<element::Int8>(shape, {-3});
    auto A = make_shared<op::ParameterizedConstant<element::Int8>>(shape, t);
    auto rt = make_shared<TensorViewType>(element::Int8::element_type(), shape);
    auto f = make_shared<Function>(A, rt, op::Parameters{});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_primary_tensor_view(element::Int8::element_type(), shape);

    cf->call({}, {result});
    ASSERT_EQ((vector<int8_t>{-3}), result->get_vector<int8_t>());
}

TEST(${BACKEND_NAME}, scalar_parameterized_constant_int32)
{
    auto shape = Shape{};
    auto t = runtime::make_tensor<element::Int32>(shape, {-3});
    auto A = make_shared<op::ParameterizedConstant<element::Int32>>(shape, t);
    auto rt = make_shared<TensorViewType>(element::Int32::element_type(), shape);
    auto f = make_shared<Function>(A, rt, op::Parameters{});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_primary_tensor_view(element::Int32::element_type(), shape);

    cf->call({}, {result});
    ASSERT_EQ((vector<int32_t>{-3}), result->get_vector<int32_t>());
}

TEST(${BACKEND_NAME}, scalar_parameterized_constant_int64)
{
    auto shape = Shape{};
    auto t = runtime::make_tensor<element::Int64>(shape, {-3});
    auto A = make_shared<op::ParameterizedConstant<element::Int64>>(shape, t);
    auto rt = make_shared<TensorViewType>(element::Int64::element_type(), shape);
    auto f = make_shared<Function>(A, rt, op::Parameters{});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_primary_tensor_view(element::Int64::element_type(), shape);

    cf->call({}, {result});
    ASSERT_EQ((vector<int64_t>{-3}), result->get_vector<int64_t>());
}

TEST(${BACKEND_NAME}, scalar_parameterized_constant_uint8)
{
    auto shape = Shape{};
    auto t = runtime::make_tensor<element::UInt8>(shape, {3});
    auto A = make_shared<op::ParameterizedConstant<element::UInt8>>(shape, t);
    auto rt = make_shared<TensorViewType>(element::UInt8::element_type(), shape);
    auto f = make_shared<Function>(A, rt, op::Parameters{});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_primary_tensor_view(element::UInt8::element_type(), shape);

    cf->call({}, {result});
    ASSERT_EQ((vector<uint8_t>{3}), result->get_vector<uint8_t>());
}

TEST(${BACKEND_NAME}, scalar_parameterized_constant_uint32)
{
    auto shape = Shape{};
    auto t = runtime::make_tensor<element::UInt32>(shape, {3});
    auto A = make_shared<op::ParameterizedConstant<element::UInt32>>(shape, t);
    auto rt = make_shared<TensorViewType>(element::UInt32::element_type(), shape);
    auto f = make_shared<Function>(A, rt, op::Parameters{});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_primary_tensor_view(element::UInt32::element_type(), shape);

    cf->call({}, {result});
    ASSERT_EQ((vector<uint32_t>{3}), result->get_vector<uint32_t>());
}

TEST(${BACKEND_NAME}, scalar_parameterized_constant_uint64)
{
    auto shape = Shape{};
    auto t = runtime::make_tensor<element::UInt64>(shape, {3});
    auto A = make_shared<op::ParameterizedConstant<element::UInt64>>(shape, t);
    auto rt = make_shared<TensorViewType>(element::UInt64::element_type(), shape);
    auto f = make_shared<Function>(A, rt, op::Parameters{});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_primary_tensor_view(element::UInt64::element_type(), shape);

    cf->call({}, {result});
    ASSERT_EQ((vector<uint64_t>{3}), result->get_vector<uint64_t>());
}

TEST(${BACKEND_NAME}, tensor_constant)
{
    auto shape = Shape{2, 2, 2};
    auto t = runtime::make_tensor<element::Float32>(shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto A = make_shared<op::ParameterizedConstant<element::Float32>>(shape, t);
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(A, rt, op::Parameters{});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    cf->call({}, {result});
    ASSERT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, tensor_constant_with_op)
{
    auto shape = Shape{2, 2, 2};
    auto t = runtime::make_tensor<element::Float32>(shape, {-1, 2, 3, -4, 5, -6, -7, 8});
    auto A = make_shared<op::ParameterizedConstant<element::Float32>>(shape, t);
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Abs>(A), rt, op::Parameters{});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    cf->call({}, {result});
    ASSERT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, function_call)
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
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(g);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    auto x = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(x, vector<float>{1, 2, 3, 4});
    auto y = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(y, vector<float>{5, 6, 7, 8});
    auto z = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(z, vector<float>{9, 10, 11, 12});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    cf->call({x, y, z}, {result});
    ASSERT_EQ((vector<float>{108, 160, 220, 288}), result->get_vector<float>());

    cf->call({y, x, z}, {result});
    ASSERT_EQ((vector<float>{108, 160, 220, 288}), result->get_vector<float>());

    cf->call({x, z, y}, {result});
    ASSERT_EQ((vector<float>{100, 144, 196, 256}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, broadcast_scalar_vector)
{
    auto shape_a = Shape{};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_r = Shape{4};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto f = make_shared<Function>(
        make_shared<op::Broadcast>(A, shape_r, AxisSet{0}), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{6});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{6, 6, 6, 6}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, broadcast_scalar_matrix)
{
    auto shape_a = Shape{};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_r = Shape{2, 2};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto f = make_shared<Function>(
        make_shared<op::Broadcast>(A, shape_r, AxisSet{0, 1}), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{6});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{6, 6, 6, 6}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, broadcast_scalar_tensor)
{
    auto shape_a = Shape{};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_r = Shape{2, 2, 2};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto f = make_shared<Function>(
        make_shared<op::Broadcast>(A, shape_r, AxisSet{0, 1, 2}), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{6});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{6, 6, 6, 6, 6, 6, 6, 6}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, broadcast_trivial)
{
    auto shape = Shape{2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(
        make_shared<op::Broadcast>(A, shape, AxisSet{}), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{2, 4, 6, 8, 16, 32, 64, 128});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{2, 4, 6, 8, 16, 32, 64, 128}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, broadcast_vector_colwise)
{
    auto shape_a = Shape{3};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_r = Shape{3, 4};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto f = make_shared<Function>(
        make_shared<op::Broadcast>(A, shape_r, AxisSet{1}), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{1, 2, 3});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, broadcast_vector_rowwise)
{
    auto shape_a = Shape{4};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_r = Shape{3, 4};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto f = make_shared<Function>(
        make_shared<op::Broadcast>(A, shape_r, AxisSet{0}), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, broadcast_vector_rowwise_int64)
{
    auto shape_a = Shape{4};
    auto A = make_shared<op::Parameter>(element::Int64::element_type(), shape_a);
    auto shape_r = Shape{3, 4};
    auto rt = make_shared<TensorViewType>(element::Int64::element_type(), shape_r);
    auto f = make_shared<Function>(
        make_shared<op::Broadcast>(A, shape_r, AxisSet{0}), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Int64::element_type(), shape_a);
    copy_data(a, vector<element::Int64::type>{1, 2, 3, 4});
    auto result = backend->make_primary_tensor_view(element::Int64::element_type(), shape_r);

    cf->call({a}, {result});
    ASSERT_EQ((vector<element::Int64::type>{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}),
              result->get_vector<element::Int64::type>());
}

TEST(DISABLED_${BACKEND_NAME}, broadcast_matrix_0)
{
    auto shape_a = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_r = Shape{2, 2, 2};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto f = make_shared<Function>(
        make_shared<op::Broadcast>(A, shape_r, AxisSet{0}), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<element::Float32::type>{1, 2, 3, 4});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a}, {result});
    ASSERT_EQ((vector<element::Float32::type>{1, 2, 3, 4, 1, 2, 3, 4}),
              result->get_vector<element::Float32::type>());
}

TEST(DISABLED_${BACKEND_NAME}, broadcast_matrix_1)
{
    auto shape_a = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_r = Shape{2, 2, 2};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto f = make_shared<Function>(
        make_shared<op::Broadcast>(A, shape_r, AxisSet{1}), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<element::Float32::type>{1, 2, 3, 4});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a}, {result});
    ASSERT_EQ((vector<element::Float32::type>{1, 2, 1, 2, 3, 4, 3, 4}),
              result->get_vector<element::Float32::type>());
}

TEST(DISABLED_${BACKEND_NAME}, broadcast_matrix_2)
{
    auto shape_a = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_r = Shape{2, 2, 2};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto f = make_shared<Function>(
        make_shared<op::Broadcast>(A, shape_r, AxisSet{2}), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<element::Float32::type>{1, 2, 3, 4});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a}, {result});
    ASSERT_EQ((vector<element::Float32::type>{1, 1, 2, 2, 3, 3, 4, 4}),
              result->get_vector<element::Float32::type>());
}

TEST(${BACKEND_NAME}, convert_int32_float32)
{
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Int32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(
        make_shared<op::Convert>(A, element::Float32::element_type()), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Int32::element_type(), shape);
    copy_data(a, vector<element::Int32::type>{1, 2, 3, 4});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    cf->call({a}, {result});
    ASSERT_EQ((vector<element::Float32::type>{1, 2, 3, 4}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, convert_int32_bool)
{
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Int32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Bool::element_type(), shape);
    auto f = make_shared<Function>(
        make_shared<op::Convert>(A, element::Bool::element_type()), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Int32::element_type(), shape);
    copy_data(a, vector<element::Int32::type>{1, 2, 3, 4});
    auto result = backend->make_primary_tensor_view(element::Bool::element_type(), shape);

    cf->call({a}, {result});
    ASSERT_EQ((vector<element::Bool::type>{1, 2, 3, 4}), result->get_vector<element::Bool::type>());
}

TEST(${BACKEND_NAME}, convert_float32_bool)
{
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Bool::element_type(), shape);
    auto f = make_shared<Function>(
        make_shared<op::Convert>(A, element::Bool::element_type()), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<element::Float32::type>{1, 2, 3, 4});
    auto result = backend->make_primary_tensor_view(element::Bool::element_type(), shape);

    cf->call({a}, {result});
    ASSERT_EQ((vector<element::Bool::type>{1, 2, 3, 4}), result->get_vector<element::Bool::type>());
}

// Trivial case with no reduction axes.
TEST(${BACKEND_NAME}, reduce_trivial)
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

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(g);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(b, vector<float>{0, 0, 0, 0});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<float>{1, 2, 3, 4}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, reduce_to_scalar)
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

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(g);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), Shape{});
    copy_data(b, vector<float>{0});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), Shape{});

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<float>{10}), result->get_vector<float>());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    ASSERT_EQ((vector<float>{1, 2, 3, 4}), a->get_vector<float>());
    ASSERT_EQ((vector<float>{0}), b->get_vector<float>());
}

TEST(${BACKEND_NAME}, reduce_matrix_columns)
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

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(g);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), Shape{});
    copy_data(b, vector<float>{0});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_rt);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<float>{9, 12}), result->get_vector<float>());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    ASSERT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), a->get_vector<float>());
    ASSERT_EQ((vector<float>{0}), b->get_vector<float>());
}

TEST(${BACKEND_NAME}, reduce_matrix_rows)
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

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(g);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), Shape{});
    copy_data(b, vector<float>{0});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_rt);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<float>{3, 7, 11}), result->get_vector<float>());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    ASSERT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), a->get_vector<float>());
    ASSERT_EQ((vector<float>{0}), b->get_vector<float>());
}

TEST(${BACKEND_NAME}, reduce_matrix_rows_zero)
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

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(g);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), Shape{});
    copy_data(b, vector<float>{66});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_rt);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<float>{66, 66, 66}), result->get_vector<float>());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    ASSERT_EQ((vector<float>{}), a->get_vector<float>());
    ASSERT_EQ((vector<float>{66}), b->get_vector<float>());
}

TEST(${BACKEND_NAME}, reduce_matrix_cols_zero)
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

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(g);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), Shape{});
    copy_data(b, vector<float>{77});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_rt);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<float>{77, 77}), result->get_vector<float>());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    ASSERT_EQ((vector<float>{}), a->get_vector<float>());
    ASSERT_EQ((vector<float>{77}), b->get_vector<float>());
}

TEST(${BACKEND_NAME}, reduce_vector_zero)
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

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(g);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), Shape{});
    copy_data(b, vector<float>{88});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_rt);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<float>{88}), result->get_vector<float>());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    ASSERT_EQ((vector<float>{}), a->get_vector<float>());
    ASSERT_EQ((vector<float>{88}), b->get_vector<float>());
}

TEST(${BACKEND_NAME}, reduce_matrix_to_scalar_zero_by_zero)
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

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(g);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), Shape{});
    copy_data(b, vector<float>{99});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_rt);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<float>{99}), result->get_vector<float>());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    ASSERT_EQ((vector<float>{}), a->get_vector<float>());
    ASSERT_EQ((vector<float>{99}), b->get_vector<float>());
}

TEST(${BACKEND_NAME}, reduce_3d_to_vector)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x*y).
    auto f_A = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
    auto f_B = make_shared<op::Parameter>(element::Float32::element_type(), Shape{});
    auto f_rt = make_shared<TensorViewType>(element::Float32::element_type(), Shape{});
    auto f =
        make_shared<Function>(make_shared<op::Multiply>(f_A, f_B), f_rt, op::Parameters{f_A, f_B});

    auto shape_a = Shape{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_b = Shape{};
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape_b);
    auto shape_rt = Shape{3};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_rt);
    auto g = make_shared<Function>(
        make_shared<op::Reduce>(A, B, f, AxisSet{0, 1}), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(g);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape_b);
    copy_data(b, vector<float>{1});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_rt);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<float>{1.0f * 10.0f * 19.0f * 4.0f * 13.0f * 22.0f * 7.0f * 16.0f * 25.0f,
                             2.0f * 11.0f * 20.0f * 5.0f * 14.0f * 23.0f * 8.0f * 17.0f * 26.0f,
                             3.0f * 12.0f * 21.0f * 6.0f * 15.0f * 24.0f * 9.0f * 18.0f * 27.0f}),
              result->get_vector<float>());
}

TEST(${BACKEND_NAME}, reshape_t2v_012)
{
    auto shape_a = Shape{2, 2, 3};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_r = Shape{12};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Reshape>(A, AxisVector{0, 1, 2}, shape_r);
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, reshape_t2s_012)
{
    auto shape_a = Shape{1, 1, 1};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_r = Shape{};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Reshape>(A, AxisVector{0, 1, 2}, shape_r);
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{6});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{6}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, reshape_t2s_120)
{
    auto shape_a = Shape{1, 1, 1};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_r = Shape{};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Reshape>(A, AxisVector{1, 2, 0}, shape_r);
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{6});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{6}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, reshape_s2t)
{
    auto shape_a = Shape{};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_r = Shape{1, 1, 1, 1, 1, 1};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Reshape>(A, AxisVector{}, shape_r);
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{42});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{42}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, reshape_v2m_col)
{
    auto shape_a = Shape{3};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_r = Shape{3, 1};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Reshape>(A, AxisVector{0}, shape_r);
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{1, 2, 3});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{1, 2, 3}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, reshape_v2m_row)
{
    auto shape_a = Shape{3};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_r = Shape{1, 3};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Reshape>(A, AxisVector{0}, shape_r);
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{1, 2, 3});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{1, 2, 3}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, reshape_v2t_middle)
{
    auto shape_a = Shape{3};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_r = Shape{1, 3, 1};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Reshape>(A, AxisVector{0}, shape_r);
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{1, 2, 3});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{1, 2, 3}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, reshape_m2m_same)
{
    auto shape_a = Shape{3, 3};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_r = Shape{3, 3};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Reshape>(A, AxisVector{0, 1}, shape_r);
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, reshape_m2m_transpose)
{
    auto shape_a = Shape{3, 3};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_r = Shape{3, 3};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Reshape>(A, AxisVector{1, 0}, shape_r);
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{1, 4, 7, 2, 5, 8, 3, 6, 9}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, reshape_m2m_dim_change_transpose)
{
    auto shape_a = Shape{3, 2};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_r = Shape{2, 3};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Reshape>(A, AxisVector{1, 0}, shape_r);
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{1, 3, 5, 2, 4, 6}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, sin)
{
    auto shape = Shape{6};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto result_type = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Sin>(A), result_type, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    float pi = acosf(-1);
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    vector<float> input{pi / 2, 0.0f, -0.0f, pi / 6, -pi, pi};
    copy_data(a, input);
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return sinf(x); });

    cf->call({a}, {result});
    ASSERT_EQ(input, result->get_vector<float>());
}

TEST(${BACKEND_NAME}, cos)
{
    auto shape = Shape{6};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto result_type = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Cos>(A), result_type, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    float pi = acosf(-1);
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    vector<float> input{pi / 2, 0.0f, -0.0f, pi / 3, -pi, pi};
    copy_data(a, input);
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return cosf(x); });

    cf->call({a}, {result});
    ASSERT_EQ(input, result->get_vector<float>());
}

TEST(${BACKEND_NAME}, tan)
{
    auto shape = Shape{6};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto result_type = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Tan>(A), result_type, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    float pi = acosf(-1);
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    vector<float> input{pi / 4, 0.0f, -0.0f, 7 * pi / 4, 3 * pi / 4, 5 * pi / 4};
    copy_data(a, input);
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return tanf(x); });

    cf->call({a}, {result});
    ASSERT_EQ(input, result->get_vector<float>());
}

TEST(${BACKEND_NAME}, asin)
{
    auto shape = Shape{6};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto result_type = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Asin>(A), result_type, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 0.5f, -0.5f};
    copy_data(a, input);
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return asinf(x); });

    cf->call({a}, {result});
    ASSERT_EQ(input, result->get_vector<float>());
}

TEST(${BACKEND_NAME}, acos)
{
    auto shape = Shape{6};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto result_type = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Acos>(A), result_type, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 0.5f, -0.5f};
    copy_data(a, input);
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return acosf(x); });

    cf->call({a}, {result});
    ASSERT_EQ(input, result->get_vector<float>());
}

TEST(${BACKEND_NAME}, atan)
{
    auto shape = Shape{6};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto result_type = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Atan>(A), result_type, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 0.5f, -0.5f};
    copy_data(a, input);
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return atanf(x); });

    cf->call({a}, {result});
    ASSERT_EQ(input, result->get_vector<float>());
}

TEST(${BACKEND_NAME}, sinh)
{
    auto shape = Shape{6};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto result_type = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Sinh>(A), result_type, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 5.0f, -5.0f};
    copy_data(a, input);
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return sinhf(x); });

    cf->call({a}, {result});
    ASSERT_EQ(input, result->get_vector<float>());
}

TEST(${BACKEND_NAME}, cosh)
{
    auto shape = Shape{6};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto result_type = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Cosh>(A), result_type, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 5.0f, -5.0f};
    copy_data(a, input);
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return coshf(x); });

    cf->call({a}, {result});
    ASSERT_EQ(input, result->get_vector<float>());
}

TEST(${BACKEND_NAME}, tanh)
{
    auto shape = Shape{6};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto result_type = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Tanh>(A), result_type, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    vector<float> input{1.0f, 0.0f, -0.0f, -1.0f, 0.5f, -0.5f};
    copy_data(a, input);
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    std::transform(
        input.begin(), input.end(), input.begin(), [](float x) -> float { return tanhf(x); });

    cf->call({a}, {result});
    ASSERT_EQ(input, result->get_vector<float>());
}

TEST(${BACKEND_NAME}, exp)
{
    auto shape = Shape{8};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto result_type = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Exp>(A), result_type, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{-4, -3, -2, -1, 0, 1, 2, 3});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    cf->call({a}, {result});
    ASSERT_EQ(
        (vector<float>{expf(-4), expf(-3), expf(-2), expf(-1), expf(0), expf(1), expf(2), expf(3)}),
        result->get_vector<float>());
}

TEST(${BACKEND_NAME}, slice_scalar)
{
    auto shape_a = Shape{};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_r = Shape{};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Slice>(A, Coordinate{}, Coordinate{});
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{312});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{312}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, slice_matrix)
{
    auto shape_a = Shape{4, 4};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_r = Shape{3, 2};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Slice>(A, Coordinate{0, 1}, Coordinate{3, 3});
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{2, 3, 6, 7, 10, 11}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, slice_vector)
{
    auto shape_a = Shape{16};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_r = Shape{12};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Slice>(A, Coordinate{2}, Coordinate{14});
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, slice_matrix_strided)
{
    auto shape_a = Shape{4, 4};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_r = Shape{2, 2};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Slice>(A, Coordinate{1, 0}, Coordinate{4, 4}, Strides{2, 3});
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{4, 7, 12, 15}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, slice_3d)
{
    auto shape_a = Shape{4, 4, 4};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_r = Shape{2, 2, 2};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Slice>(A, Coordinate{1, 1, 1}, Coordinate{3, 3, 3});
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{21, 22, 25, 26, 37, 38, 41, 42}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, slice_3d_strided)
{
    auto shape_a = Shape{4, 4, 4};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_r = Shape{2, 2, 2};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Slice>(A, Coordinate{0, 0, 0}, Coordinate{4, 4, 4}, Strides{2, 2, 2});
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{0, 2, 8, 10, 32, 34, 40, 42}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, slice_3d_strided_different_strides)
{
    auto shape_a = Shape{4, 4, 4};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_r = Shape{2, 2, 2};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Slice>(A, Coordinate{0, 0, 0}, Coordinate{4, 4, 4}, Strides{2, 2, 3});
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{0, 3, 8, 11, 32, 35, 40, 43}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, scalar_constant_float32)
{
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), Shape{});
    auto r = make_shared<op::Constant>(element::Float32::element_type(), Shape{}, "4.8");
    auto f = make_shared<Function>(r, rt, op::Parameters{});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), Shape{});

    cf->call({}, {result});
    ASSERT_EQ(vector<float>{std::strtof("4.8", NULL)}, result->get_vector<float>());
}

TEST(${BACKEND_NAME}, scalar_constant_int64)
{
    auto rt = make_shared<TensorViewType>(element::Int64::element_type(), Shape{});
    auto r = make_shared<op::Constant>(element::Int64::element_type(), Shape{}, "2112");
    auto f = make_shared<Function>(r, rt, op::Parameters{});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_primary_tensor_view(element::Int64::element_type(), Shape{});

    cf->call({}, {result});
    ASSERT_EQ(vector<element::Int64::type>{std::strtol("2112", NULL, 10)},
              result->get_vector<element::Int64::type>());
}

TEST(${BACKEND_NAME}, tensor_constant_float32)
{
    auto shape = Shape{2, 2};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto r = make_shared<op::Constant>(element::Float32::element_type(),
                                       shape,
                                       std::vector<std::string>{"4.8", "4.7", "-5.3", "0"});
    auto f = make_shared<Function>(r, rt, op::Parameters{});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    cf->call({}, {result});
    ASSERT_EQ((vector<float>{std::strtof("4.8", NULL),
                             std::strtof("4.7", NULL),
                             std::strtof("-5.3", NULL),
                             std::strtof("0", NULL)}),
              result->get_vector<float>());
}

TEST(${BACKEND_NAME}, tensor_constant_int64)
{
    auto shape = Shape{2, 2};
    auto rt = make_shared<TensorViewType>(element::Int64::element_type(), shape);
    auto r = make_shared<op::Constant>(element::Int64::element_type(),
                                       shape,
                                       std::vector<std::string>{"2112", "1848", "1776", "1964"});
    auto f = make_shared<Function>(r, rt, op::Parameters{});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_primary_tensor_view(element::Int64::element_type(), shape);

    cf->call({}, {result});
    ASSERT_EQ((vector<element::Int64::type>{std::strtol("2112", NULL, 10),
                                            std::strtol("1848", NULL, 10),
                                            std::strtol("1776", NULL, 10),
                                            std::strtol("1964", NULL, 10)}),
              result->get_vector<element::Int64::type>());
}

// Trivial case with no summed axes.
TEST(${BACKEND_NAME}, sum_trivial)
{
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{}), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{1, 2, 3, 4}), result->get_vector<float>());
}

// Failure has been reported at 5D for some reason
TEST(${BACKEND_NAME}, sum_trivial_5d)
{
    auto shape = Shape{2, 2, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{}), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
              result->get_vector<float>());
}

TEST(${BACKEND_NAME}, sum_to_scalar)
{
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), Shape{});
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0, 1}), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), Shape{});

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{10}), result->get_vector<float>());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    ASSERT_EQ((vector<float>{1, 2, 3, 4}), a->get_vector<float>());
}

TEST(${BACKEND_NAME}, sum_matrix_columns)
{
    auto shape_a = Shape{3, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_rt = Shape{2};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_rt);
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0}), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_rt);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{9, 12}), result->get_vector<float>());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    ASSERT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), a->get_vector<float>());
}

TEST(${BACKEND_NAME}, sum_matrix_rows)
{
    auto shape_a = Shape{3, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_rt = Shape{3};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_rt);
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{1}), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_rt);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{3, 7, 11}), result->get_vector<float>());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    ASSERT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), a->get_vector<float>());
}

TEST(${BACKEND_NAME}, sum_matrix_rows_zero)
{
    auto shape_a = Shape{3, 0};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_rt = Shape{3};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_rt);
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{1}), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_rt);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{0, 0, 0}), result->get_vector<float>());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    ASSERT_EQ((vector<float>{}), a->get_vector<float>());
}

TEST(${BACKEND_NAME}, sum_matrix_cols_zero)
{
    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    auto shape_a = Shape{0, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_rt = Shape{2};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_rt);
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0}), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_rt);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{0, 0}), result->get_vector<float>());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    ASSERT_EQ((vector<float>{}), a->get_vector<float>());
}

TEST(${BACKEND_NAME}, sum_vector_zero)
{
    auto shape_a = Shape{0};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_rt = Shape{};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_rt);
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0}), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_rt);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{0}), result->get_vector<float>());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    ASSERT_EQ((vector<float>{}), a->get_vector<float>());
}

TEST(${BACKEND_NAME}, sum_matrix_to_scalar_zero_by_zero)
{
    auto shape_a = Shape{0, 0};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_rt = Shape{};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_rt);
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0, 1}), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_rt);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{0}), result->get_vector<float>());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    ASSERT_EQ((vector<float>{}), a->get_vector<float>());
}

TEST(${BACKEND_NAME}, sum_3d_to_matrix_most_sig)
{
    auto shape_a = Shape{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_rt = Shape{3, 3};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_rt);
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0}), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_rt);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{1 + 10 + 19,
                             2 + 11 + 20,
                             3 + 12 + 21,
                             4 + 13 + 22,
                             5 + 14 + 23,
                             6 + 15 + 24,
                             7 + 16 + 25,
                             8 + 17 + 26,
                             9 + 18 + 27}),
              result->get_vector<float>());
}

TEST(${BACKEND_NAME}, sum_3d_to_matrix_least_sig)
{
    auto shape_a = Shape{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_rt = Shape{3, 3};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_rt);
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{2}), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_rt);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{1 + 2 + 3,
                             4 + 5 + 6,
                             7 + 8 + 9,
                             10 + 11 + 12,
                             13 + 14 + 15,
                             16 + 17 + 18,
                             19 + 20 + 21,
                             22 + 23 + 24,
                             25 + 26 + 27}),
              result->get_vector<float>());
}

TEST(${BACKEND_NAME}, sum_3d_to_vector)
{
    auto shape_a = Shape{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_rt = Shape{3};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_rt);
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0, 1}), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_rt);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{1 + 10 + 19 + 4 + 13 + 22 + 7 + 16 + 25,
                             2 + 11 + 20 + 5 + 14 + 23 + 8 + 17 + 26,
                             3 + 12 + 21 + 6 + 15 + 24 + 9 + 18 + 27}),
              result->get_vector<float>());
}

TEST(${BACKEND_NAME}, sum_3d_to_scalar)
{
    auto shape_a = Shape{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_rt = Shape{};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_rt);
    auto f =
        make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0, 1, 2}), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_rt);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{1 + 10 + 19 + 4 + 13 + 22 + 7 + 16 + 25 + 2 + 11 + 20 + 5 + 14 + 23 +
                             8 + 17 + 26 + 3 + 12 + 21 + 6 + 15 + 24 + 9 + 18 + 27}),
              result->get_vector<float>());
}

TEST(${BACKEND_NAME}, sum_3d_eliminate_zero_dim)
{
    auto shape_a = Shape{3, 0, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_rt = Shape{3, 2};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_rt);
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{1}), rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_rt);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the right value.
    copy_data(result, vector<float>{2112, 2112, 2112, 2112, 2112, 2112});

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{0, 0, 0, 0, 0, 0}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, sign)
{
    auto shape = Shape{2, 3};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto result_type = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Sign>(A), result_type, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{1, -2, 0, -4.8f, 4.8f, -0.0});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{1, -1, 0, -1, 1, 0}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, power)
{
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Power>(A, B), rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{1, 2, 3, 5});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(b, vector<float>{2, 0, 6, 3});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<float>{1, 1, 729, 125}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, constant_equality_bool)
{
    auto shape = Shape{4};
    // auto A = make_shared<op::Parameter>(element::Bool::element_type(), shape);
    // auto B = make_shared<op::Parameter>(element::Bool::element_type(), shape);
    // auto result_type = make_shared<TensorViewType>(element::Bool::element_type(), shape);
    // auto f = make_shared<Function>(make_shared<op::Equal>(A, B), result_type, op::Parameters{A, B});

    auto a = runtime::make_tensor<element::Bool>(shape, {true, false, true, false});
    auto A = make_shared<op::ParameterizedConstant<element::Bool>>(shape, a);
    auto b = runtime::make_tensor<element::Bool>(shape, {true, true, true, true});
    auto B = make_shared<op::ParameterizedConstant<element::Bool>>(shape, b);
    auto rt = make_shared<TensorViewType>(element::Bool::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Equal>(A, B), rt, op::Parameters{});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_primary_tensor_view(element::Bool::element_type(), shape);

    cf->call({}, {result});
    ASSERT_EQ((vector<char>{true, false, true, false}), result->get_vector<char>());
}

TEST(${BACKEND_NAME}, sqrt)
{
    auto shape = Shape{2, 3};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto result_type = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Sqrt>(A), result_type, op::Parameters{A});

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{16, 4, 81, 100, 10000, 0});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    cf->call({a}, {result});
    ASSERT_EQ((vector<float>{4, 2, 9, 10, 100, 0}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, replace_slice_scalar)
{
    auto shape_a = Shape{};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_b = Shape{};
    auto B = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_b));
    auto shape_r = Shape{};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::ReplaceSlice>(A, B, Coordinate{}, Coordinate{});
    auto f = make_shared<Function>(r, rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{312});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape_b);
    copy_data(b, vector<float>{808});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<float>{808}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, replace_slice_matrix)
{
    auto shape_a = Shape{4, 4};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_b = Shape{3, 2};
    auto B = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_b));
    auto shape_r = Shape{4, 4};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::ReplaceSlice>(A, B, Coordinate{0, 1}, Coordinate{3, 3});
    auto f = make_shared<Function>(r, rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape_b);
    copy_data(b, vector<float>{102, 103, 106, 107, 110, 111});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<float>{1, 102, 103, 4, 5, 106, 107, 8, 9, 110, 111, 12, 13, 14, 15, 16}),
              result->get_vector<float>());
}

TEST(${BACKEND_NAME}, replace_slice_vector)
{
    auto shape_a = Shape{16};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_b = Shape{12};
    auto B = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_b));
    auto shape_r = Shape{16};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::ReplaceSlice>(A, B, Coordinate{2}, Coordinate{14});
    auto f = make_shared<Function>(r, rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape_b);
    copy_data(b, vector<float>{102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a, b}, {result});
    ASSERT_EQ(
        (vector<float>{0, 1, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 14, 15}),
        result->get_vector<float>());
}

TEST(${BACKEND_NAME}, one_hot_scalar_2_in_3)
{
    auto shape_a = Shape{};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Int32::element_type(), shape_a));
    auto shape_r = Shape{3};
    auto rt = make_shared<TensorViewType>(element::Int32::element_type(), shape_r);
    auto r = make_shared<op::OneHot>(A, Shape{3}, 0);
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Int32::element_type(), shape_a);
    copy_data(a, vector<int32_t>{2});
    auto result = backend->make_primary_tensor_view(element::Int32::element_type(), shape_r);

    cf->call({a}, {result});
    ASSERT_EQ((vector<int32_t>{0, 0, 1}), result->get_vector<int32_t>());
}

TEST(${BACKEND_NAME}, one_hot_scalar_1_in_3)
{
    auto shape_a = Shape{};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Int32::element_type(), shape_a));
    auto shape_r = Shape{3};
    auto rt = make_shared<TensorViewType>(element::Int32::element_type(), shape_r);
    auto r = make_shared<op::OneHot>(A, Shape{3}, 0);
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Int32::element_type(), shape_a);
    copy_data(a, vector<int32_t>{1});
    auto result = backend->make_primary_tensor_view(element::Int32::element_type(), shape_r);

    cf->call({a}, {result});
    ASSERT_EQ((vector<int32_t>{0, 1, 0}), result->get_vector<int32_t>());
}

TEST(${BACKEND_NAME}, one_hot_scalar_0_in_3)
{
    auto shape_a = Shape{};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Int32::element_type(), shape_a));
    auto shape_r = Shape{3};
    auto rt = make_shared<TensorViewType>(element::Int32::element_type(), shape_r);
    auto r = make_shared<op::OneHot>(A, Shape{3}, 0);
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Int32::element_type(), shape_a);
    copy_data(a, vector<int32_t>{0});
    auto result = backend->make_primary_tensor_view(element::Int32::element_type(), shape_r);

    cf->call({a}, {result});
    ASSERT_EQ((vector<int32_t>{1, 0, 0}), result->get_vector<int32_t>());
}

TEST(${BACKEND_NAME}, one_hot_scalar_fp_nonint_in_3)
{
    auto shape_a = Shape{};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_r = Shape{3};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::OneHot>(A, Shape{3}, 0);
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{1.1f});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    EXPECT_THROW({ cf->call({a}, {result}); }, std::range_error);
}

TEST(${BACKEND_NAME}, one_hot_scalar_oob_in_3)
{
    auto shape_a = Shape{};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Int32::element_type(), shape_a));
    auto shape_r = Shape{3};
    auto rt = make_shared<TensorViewType>(element::Int32::element_type(), shape_r);
    auto r = make_shared<op::OneHot>(A, Shape{3}, 0);
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Int32::element_type(), shape_a);
    copy_data(a, vector<int32_t>{3000000});
    auto result = backend->make_primary_tensor_view(element::Int32::element_type(), shape_r);

    EXPECT_THROW({ cf->call({a}, {result}); }, std::range_error);
}

TEST(${BACKEND_NAME}, one_hot_vector_0)
{
    auto shape_a = Shape{8};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Int32::element_type(), shape_a));
    auto shape_r = Shape{3, 8};
    auto rt = make_shared<TensorViewType>(element::Int32::element_type(), shape_r);
    auto r = make_shared<op::OneHot>(A, Shape{3, 8}, 0);
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Int32::element_type(), shape_a);
    copy_data(a, vector<int32_t>{2, 1, 0, 0, 2, 2, 1, 0});
    auto result = backend->make_primary_tensor_view(element::Int32::element_type(), shape_r);

    cf->call({a}, {result});
    ASSERT_EQ(
        (vector<int32_t>{0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0}),
        result->get_vector<int32_t>());
}

TEST(${BACKEND_NAME}, one_hot_vector_1)
{
    auto shape_a = Shape{8};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Int32::element_type(), shape_a));
    auto shape_r = Shape{8, 3};
    auto rt = make_shared<TensorViewType>(element::Int32::element_type(), shape_r);
    auto r = make_shared<op::OneHot>(A, Shape{8, 3}, 1);
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Int32::element_type(), shape_a);
    copy_data(a, vector<int32_t>{2, 1, 0, 0, 2, 2, 1, 0});
    auto result = backend->make_primary_tensor_view(element::Int32::element_type(), shape_r);

    cf->call({a}, {result});
    ASSERT_EQ(
        (vector<int32_t>{0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0}),
        result->get_vector<int32_t>());
}

TEST(${BACKEND_NAME}, one_hot_vector_1_barely_oob)
{
    auto shape_a = Shape{8};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Int32::element_type(), shape_a));
    auto shape_r = Shape{8, 3};
    auto rt = make_shared<TensorViewType>(element::Int32::element_type(), shape_r);
    auto r = make_shared<op::OneHot>(A, Shape{8, 3}, 1);
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Int32::element_type(), shape_a);
    copy_data(a, vector<int32_t>{2, 1, 0, 0, 3, 2, 1, 0});
    auto result = backend->make_primary_tensor_view(element::Int32::element_type(), shape_r);

    EXPECT_THROW({ cf->call({a}, {result}); }, std::range_error);
}

TEST(${BACKEND_NAME}, one_hot_vector_1_far_oob)
{
    auto shape_a = Shape{8};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Int32::element_type(), shape_a));
    auto shape_r = Shape{8, 3};
    auto rt = make_shared<TensorViewType>(element::Int32::element_type(), shape_r);
    auto r = make_shared<op::OneHot>(A, Shape{8, 3}, 1);
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Int32::element_type(), shape_a);
    copy_data(a, vector<int32_t>{2, 1, 0, 0, 3000000, 2, 1, 0});
    auto result = backend->make_primary_tensor_view(element::Int32::element_type(), shape_r);

    EXPECT_THROW({ cf->call({a}, {result}); }, std::range_error);
}

// This test is disabled because it won't yet work on the IA backend, but it does work with
// the de-Eigenized kernel on NGVM.
//
// Test if you like with:
//
//    private-ngraph-cpp/build$ test/unit-test \
//                                 --gtest_filter='DISABLED_NGVM.one_hot_matrix_0' \
//                                 --gtest_also_run_disabled_tests
TEST(DISABLED_${BACKEND_NAME}, one_hot_matrix_0)
{
    auto shape_a = Shape{3, 3};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Int32::element_type(), shape_a));
    auto shape_r = Shape{3, 3, 3};
    auto rt = make_shared<TensorViewType>(element::Int32::element_type(), shape_r);
    auto r = make_shared<op::OneHot>(A, Shape{3, 3, 3}, 0);
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Int32::element_type(), shape_a);
    copy_data(a,
              vector<int32_t>{
                  0, 1, 1, 2, 1, 0, 0, 2, 1,
              });
    auto result = backend->make_primary_tensor_view(element::Int32::element_type(), shape_r);

    cf->call({a}, {result});
    ASSERT_EQ((vector<int32_t>{1, 0, 0, 0, 0, 1, 1, 0, 0,

                               0, 1, 1, 0, 1, 0, 0, 0, 1,

                               0, 0, 0, 1, 0, 0, 0, 1, 0}),
              result->get_vector<int32_t>());
}

TEST(${BACKEND_NAME}, one_hot_vector_1_fp)
{
    auto shape_a = Shape{8};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_r = Shape{8, 3};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::OneHot>(A, Shape{8, 3}, 1);
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{2, 1, 0, 0, 2, 2, 1, 0});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a}, {result});
    ASSERT_EQ(
        (vector<float>{0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0}),
        result->get_vector<float>());
}

TEST(${BACKEND_NAME}, one_hot_vector_1_fp_nonint)
{
    auto shape_a = Shape{8};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_r = Shape{8, 3};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::OneHot>(A, Shape{8, 3}, 1);
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{2, 1, 0, 0, 2, 2, 1.01f, 0});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    EXPECT_THROW({ cf->call({a}, {result}); }, std::range_error);
}

TEST(${BACKEND_NAME}, replace_slice_3d)
{
    auto shape_a = Shape{4, 4, 4};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_b = Shape{2, 2, 2};
    auto B = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_b));
    auto shape_r = Shape{4, 4, 4};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::ReplaceSlice>(A, B, Coordinate{1, 1, 1}, Coordinate{3, 3, 3});
    auto f = make_shared<Function>(r, rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape_b);
    copy_data(b, vector<float>{921, 922, 925, 926, 937, 938, 941, 942});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<float>{0,  1,  2,  3,  4,  5,   6,   7,  8,  9,   10,  11, 12, 13, 14, 15,

                             16, 17, 18, 19, 20, 921, 922, 23, 24, 925, 926, 27, 28, 29, 30, 31,

                             32, 33, 34, 35, 36, 937, 938, 39, 40, 941, 942, 43, 44, 45, 46, 47,

                             48, 49, 50, 51, 52, 53,  54,  55, 56, 57,  58,  59, 60, 61, 62, 63}),
              result->get_vector<float>());
}

TEST(${BACKEND_NAME}, replace_slice_3d_strided)
{
    auto shape_a = Shape{4, 4, 4};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_b = Shape{2, 2, 2};
    auto B = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_b));
    auto shape_r = Shape{4, 4, 4};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::ReplaceSlice>(
        A, B, Coordinate{0, 0, 0}, Coordinate{4, 4, 4}, Strides{2, 2, 2});
    auto f = make_shared<Function>(r, rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape_b);
    copy_data(b, vector<float>{900, 902, 908, 910, 932, 934, 940, 942});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<float>{900, 1,  902, 3,  4,  5,  6,  7,  908, 9,  910, 11, 12, 13, 14, 15,

                             16,  17, 18,  19, 20, 21, 22, 23, 24,  25, 26,  27, 28, 29, 30, 31,

                             932, 33, 934, 35, 36, 37, 38, 39, 940, 41, 942, 43, 44, 45, 46, 47,

                             48,  49, 50,  51, 52, 53, 54, 55, 56,  57, 58,  59, 60, 61, 62, 63}),
              result->get_vector<float>());
}

TEST(${BACKEND_NAME}, replace_slice_3d_strided_different_strides)
{
    auto shape_a = Shape{4, 4, 4};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_b = Shape{2, 2, 2};
    auto B = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_b));
    auto shape_r = Shape{4, 4, 4};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::ReplaceSlice>(
        A, B, Coordinate{0, 0, 0}, Coordinate{4, 4, 4}, Strides{2, 2, 3});
    auto f = make_shared<Function>(r, rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, vector<float>{0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,

                               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,

                               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,

                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape_b);
    copy_data(b, vector<float>{900, 903, 908, 911, 932, 935, 940, 943});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a, b}, {result});
    ASSERT_EQ((vector<float>{900, 1,  2,  903, 4,  5,  6,  7,  908, 9,  10, 911, 12, 13, 14, 15,

                             16,  17, 18, 19,  20, 21, 22, 23, 24,  25, 26, 27,  28, 29, 30, 31,

                             932, 33, 34, 935, 36, 37, 38, 39, 940, 41, 42, 943, 44, 45, 46, 47,

                             48,  49, 50, 51,  52, 53, 54, 55, 56,  57, 58, 59,  60, 61, 62, 63}),
              result->get_vector<float>());
}
