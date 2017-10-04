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

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;

TEST(execute, test_abc)
{
    auto shape = Shape{2, 2};
    auto A     = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B     = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto C     = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt    = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f     = make_shared<Function>((A + B) * C, rt, op::Parameters{A, B, C});

    auto transformer = runtime::Transformer::get_transformer("NGVM");
    auto external = transformer->compile(f);
    
    auto backend = transformer->allocate_backend();
    auto cf       = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a      = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *a          = vector<float>{1, 2, 3, 4};
    auto b      = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *b          = vector<float>{5, 6, 7, 8};
    auto c      = backend->make_parameterized_tensor_view<element::Float32>(shape);
    *c          = vector<float>{9, 10, 11, 12};
    auto result = backend->make_parameterized_tensor_view<element::Float32>(shape);

    (*cf)({a, b, c}, {result});
    ASSERT_EQ((vector<float>{54, 80, 110, 144}), result->get_vector());

    (*cf)({b, a, c}, {result});
    ASSERT_EQ((vector<float>{54, 80, 110, 144}), result->get_vector());

    (*cf)({a, c, b}, {result});
    ASSERT_EQ((vector<float>{50, 72, 98, 128}), result->get_vector());
}

// Same as test_abc, but using tuples for input and output
TEST(execute, test_abc_tuple)
{
    auto shape = Shape{2, 2};

    auto tensor_view_type = make_shared<TensorViewType>(element::Float32::element_type(), shape);

    auto ABC = make_shared<op::Parameter>(
        make_shared<TupleType>(ValueTypes{tensor_view_type, tensor_view_type, tensor_view_type}));

    auto A = make_shared<op::GetTupleElement>(ABC, 0);
    auto B = make_shared<op::GetTupleElement>(ABC, 1);
    auto C = make_shared<op::GetTupleElement>(ABC, 2);
    auto f = make_shared<Function>(make_shared<op::Tuple>(Nodes{(A + B) * C}),
                                   tensor_view_type,
                                   op::Parameters{ABC});

    auto external = make_shared<ngraph::runtime::ExternalFunction>(f);
    auto cf       = external->make_call_frame();

    // Create some tensors for input/output
    auto a            = ngraph::runtime::make_tensor<element::Float32>(shape);
    *a                = vector<float>{1, 2, 3, 4};
    auto b            = ngraph::runtime::make_tensor<element::Float32>(shape);
    *b                = vector<float>{5, 6, 7, 8};
    auto c            = ngraph::runtime::make_tensor<element::Float32>(shape);
    *c                = vector<float>{9, 10, 11, 12};
    auto abc          = ngraph::runtime::make_tuple({a, b, c});
    auto bac          = ngraph::runtime::make_tuple({b, a, c});
    auto acb          = ngraph::runtime::make_tuple({a, c, b});
    auto result       = ngraph::runtime::make_tensor<element::Float32>(shape);
    auto result_tuple = ngraph::runtime::make_tuple({result});

    (*cf)({abc}, {result_tuple});
    ASSERT_EQ((vector<float>{54, 80, 110, 144}), result->get_vector());

    (*cf)({bac}, {result_tuple});
    ASSERT_EQ((vector<float>{54, 80, 110, 144}), result->get_vector());

    (*cf)({acb}, {result_tuple});
    ASSERT_EQ((vector<float>{50, 72, 98, 128}), result->get_vector());
}

// Multiple retrive values
TEST(execute, test_tuple_result)
{
    auto shape         = Shape{2, 2};
    auto A             = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B             = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto C             = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto A_add_B       = make_shared<op::Add>(A, B);
    auto A_add_B_mul_C = make_shared<op::Multiply>(A_add_B, C);

    auto rt = make_shared<TupleType>(
                std::vector<shared_ptr<const ValueType>>(
                 {make_shared<TensorViewType>(element::Float32::element_type(), shape),
                  make_shared<TensorViewType>(element::Float32::element_type(), shape)}));
    auto f = make_shared<Function>(make_shared<op::Tuple>(Nodes{A_add_B, A_add_B_mul_C}),
                                   rt, op::Parameters{A, B, C});
    auto external = make_shared<ngraph::runtime::ExternalFunction>(f);
    auto cf       = external->make_call_frame();

    auto a = ngraph::runtime::make_tensor<element::Float32>(shape);
    *a     = vector<float>{1, 2, 3, 4};
    auto b = ngraph::runtime::make_tensor<element::Float32>(shape);
    *b     = vector<float>{5, 6, 7, 8};
    auto c = ngraph::runtime::make_tensor<element::Float32>(shape);
    *c     = vector<float>{9, 10, 11, 12};

    auto r0 = ngraph::runtime::make_tensor<element::Float32>(shape);
    auto r1 = ngraph::runtime::make_tensor<element::Float32>(shape);
    auto result_tuple = ngraph::runtime::make_tuple({r0, r1});

    (*cf)({a, b, c}, {result_tuple});

    ASSERT_EQ((vector<float>{6, 8, 10, 12}), r0->get_vector());
    ASSERT_EQ((vector<float>{54, 80, 110, 144}), r1->get_vector());
}

TEST(execute, test_abs)
{
    auto shape       = Shape{2, 2};
    auto A           = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto result_type = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f           = make_shared<Function>(make_shared<op::Abs>(A), result_type, op::Parameters{A});

    auto external = make_shared<ngraph::runtime::ExternalFunction>(f);
    auto cf       = external->make_call_frame();

    // Create some tensors for input/output
    auto a      = ngraph::runtime::make_tensor<element::Float32>(shape);
    *a          = vector<float>{1, -2, 0, -4.8f};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape);

    (*cf)({a}, {result});
    ASSERT_EQ((vector<float>{1, 2, 0, 4.8f}), result->get_vector());
}

TEST(execute, test_concat_matrix_colwise)
{
    auto shape_a = Shape{2, 2};
    auto A       = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_b = Shape{2, 3};
    auto B       = make_shared<op::Parameter>(element::Float32::element_type(), shape_b);
    auto shape_c = Shape{2, 3};
    auto C       = make_shared<op::Parameter>(element::Float32::element_type(), shape_c);
    auto shape_r = Shape{2, 8};
    auto rt      = make_shared<TensorViewType>(element::Float32::element_type(), Shape{2,8});
    auto f       = make_shared<Function>(make_shared<op::Concat>(Nodes{A,B,C},1), rt, op::Parameters{A,B,C});

    auto external = make_shared<ngraph::runtime::ExternalFunction>(f);
    auto cf       = external->make_call_frame();

    // Create some tensors for input/output
    auto a      = ngraph::runtime::make_tensor<element::Float32>(shape_a);
    *a          = vector<float>{2, 4, 8, 16};
    auto b      = ngraph::runtime::make_tensor<element::Float32>(shape_b);
    *b          = vector<float>{1, 2, 4, 8, 16, 32};
    auto c      = ngraph::runtime::make_tensor<element::Float32>(shape_c);
    *c          = vector<float>{2, 3, 5, 7, 11, 13};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape_r);

    (*cf)({a, b, c}, {result});
    ASSERT_EQ((vector<float>{2, 4, 1, 2, 4, 2, 3, 5, 8, 16, 8, 16, 32, 7, 11, 13}),
              result->get_vector());
}

TEST(execute, test_concat_matrix_rowwise)
{
    auto shape_a = Shape{2, 2};
    auto A       = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_b = Shape{3, 2};
    auto B       = make_shared<op::Parameter>(element::Float32::element_type(), shape_b);
    auto shape_c = Shape{3, 2};
    auto C       = make_shared<op::Parameter>(element::Float32::element_type(), shape_c);
    auto shape_r = Shape{8, 2};
    auto rt      = make_shared<TensorViewType>(element::Float32::element_type(), Shape{8,2});
    auto f       = make_shared<Function>(make_shared<op::Concat>(Nodes{A,B,C},0), rt, op::Parameters{A,B,C});

    auto external = make_shared<ngraph::runtime::ExternalFunction>(f);
    auto cf       = external->make_call_frame();

    // Create some tensors for input/output
    auto a      = ngraph::runtime::make_tensor<element::Float32>(shape_a);
    *a          = vector<float>{2, 4, 8, 16};
    auto b      = ngraph::runtime::make_tensor<element::Float32>(shape_b);
    *b          = vector<float>{1, 2, 4, 8, 16, 32};
    auto c      = ngraph::runtime::make_tensor<element::Float32>(shape_c);
    *c          = vector<float>{2, 3, 5, 7, 11, 13};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape_r);

    (*cf)({a, b, c}, {result});
    ASSERT_EQ((vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13}),
              result->get_vector());
}

TEST(execute, test_concat_vector)
{
    auto shape_a = Shape{4};
    auto A       = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_b = Shape{6};
    auto B       = make_shared<op::Parameter>(element::Float32::element_type(), shape_b);
    auto shape_c = Shape{2};
    auto C       = make_shared<op::Parameter>(element::Float32::element_type(), shape_c);
    auto shape_r = Shape{12};
    auto rt      = make_shared<TensorViewType>(element::Float32::element_type(), Shape{12});
    auto f       = make_shared<Function>(make_shared<op::Concat>(Nodes{A,B,C},0), rt, op::Parameters{A,B,C});

    auto external = make_shared<ngraph::runtime::ExternalFunction>(f);
    auto cf       = external->make_call_frame();

    // Create some tensors for input/output
    auto a      = ngraph::runtime::make_tensor<element::Float32>(shape_a);
    *a          = vector<float>{2, 4, 8, 16};
    auto b      = ngraph::runtime::make_tensor<element::Float32>(shape_b);
    *b          = vector<float>{1, 2, 4, 8, 16, 32};
    auto c      = ngraph::runtime::make_tensor<element::Float32>(shape_c);
    *c          = vector<float>{18, 19};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape_r);

    (*cf)({a, b, c}, {result});
    ASSERT_EQ((vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 18, 19}), result->get_vector());
}

TEST(execute, test_divide)
{
    auto shape = Shape{2, 2};
    auto A     = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B     = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt    = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f     = make_shared<Function>(make_shared<op::Divide>(A, B), rt, op::Parameters{A, B});

    auto external = make_shared<ngraph::runtime::ExternalFunction>(f);
    auto cf       = external->make_call_frame();

    // Create some tensors for input/output
    auto a      = ngraph::runtime::make_tensor<element::Float32>(shape);
    *a          = vector<float>{2, 4, 8, 16};
    auto b      = ngraph::runtime::make_tensor<element::Float32>(shape);
    *b          = vector<float>{1, 2, 4, 8};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<float>{2, 2, 2, 2}), result->get_vector());
}

TEST(execute, test_equal)
{
    auto shape = Shape{2, 2, 2};
    auto A     = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B     = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt    = make_shared<TensorViewType>(element::Bool::element_type(), shape);
    auto f     = make_shared<Function>(make_shared<op::Equal>(A, B), rt, op::Parameters{A, B});

    auto external = make_shared<ngraph::runtime::ExternalFunction>(f);
    auto cf       = external->make_call_frame();

    // Create some tensors for input/output
    auto a      = ngraph::runtime::make_tensor<element::Float32>(shape);
    *a          = vector<float>{1, 8, -8, 17, -0.5, 0, 1, 1};
    auto b      = ngraph::runtime::make_tensor<element::Float32>(shape);
    *b          = vector<float>{1, 8, 4, 8, 0, 0, 1, 1.5};
    auto result = ngraph::runtime::make_tensor<element::Bool>(shape);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<char>{1, 1, 0, 0, 0, 1, 1, 0}), result->get_vector());
}

TEST(execute, test_dot1d)
{
    auto shape   = Shape{4};
    auto A       = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B       = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto shape_r = Shape{1};
    auto rt      = make_shared<TensorViewType>(element::Float32::element_type(), Shape{});
    auto f       = make_shared<Function>(make_shared<op::Dot>(A,B), rt, op::Parameters{A,B});

    auto external = make_shared<ngraph::runtime::ExternalFunction>(f);
    auto cf       = external->make_call_frame();

    // Create some tensors for input/output
    auto a      = ngraph::runtime::make_tensor<element::Float32>(shape);
    *a          = vector<float>{2, 4, 8, 16};
    auto b      = ngraph::runtime::make_tensor<element::Float32>(shape);
    *b          = vector<float>{1, 2, 4, 8};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape_r);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<float>{170}), result->get_vector());
}

TEST(execute, test_dot2d)
{
    auto shape   = Shape{2, 2};
    auto A       = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B       = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto shape_r = Shape{2,2};
    auto rt      = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f       = make_shared<Function>(make_shared<op::Dot>(A,B), rt, op::Parameters{A,B});

    auto external = make_shared<ngraph::runtime::ExternalFunction>(f);
    auto cf       = external->make_call_frame();

    // Create some tensors for input/output
    auto a      = ngraph::runtime::make_tensor<element::Float32>(shape);
    *a          = vector<float>{1, 2, 3, 4};
    auto b      = ngraph::runtime::make_tensor<element::Float32>(shape);
    *b          = vector<float>{5, 6, 7, 8};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape_r);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<float>{19, 22, 43, 50}), result->get_vector());
}

TEST(execute, test_dot_scalar_tensor_arg0)
{
    auto shape_a = Shape{};
    auto shape_b = Shape{2,2,2};
    auto A       = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto B       = make_shared<op::Parameter>(element::Float32::element_type(), shape_b);
    auto rt      = make_shared<TensorViewType>(element::Float32::element_type(), shape_b);
    auto f       = make_shared<Function>(make_shared<op::Dot>(A,B), rt, op::Parameters{A,B});

    auto external = make_shared<ngraph::runtime::ExternalFunction>(f);
    auto cf       = external->make_call_frame();

    // Create some tensors for input/output
    auto a      = ngraph::runtime::make_tensor<element::Float32>(shape_a);
    *a          = vector<float>{6};
    auto b      = ngraph::runtime::make_tensor<element::Float32>(shape_b);
    *b          = vector<float>{1, 2, 3, 4, 5, 6, 7, 8};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape_b);

    (*cf)({a,b}, {result});
    ASSERT_EQ((vector<float>{6, 12, 18, 24, 30, 36, 42, 48}), result->get_vector());
}

TEST(execute, test_dot_scalar_tensor_arg1)
{
    auto shape_a = Shape{2,2,2};
    auto shape_b = Shape{};
    auto A       = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto B       = make_shared<op::Parameter>(element::Float32::element_type(), shape_b);
    auto rt      = make_shared<TensorViewType>(element::Float32::element_type(), shape_a);
    auto f       = make_shared<Function>(make_shared<op::Dot>(A,B), rt, op::Parameters{A,B});

    auto external = make_shared<ngraph::runtime::ExternalFunction>(f);
    auto cf       = external->make_call_frame();

    // Create some tensors for input/output
    auto a      = ngraph::runtime::make_tensor<element::Float32>(shape_a);
    *a          = vector<float>{1, 2, 3, 4, 5, 6, 7, 8};
    auto b      = ngraph::runtime::make_tensor<element::Float32>(shape_b);
    *b          = vector<float>{6};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape_a);

    (*cf)({a,b}, {result});
    ASSERT_EQ((vector<float>{6, 12, 18, 24, 30, 36, 42, 48}), result->get_vector());
}

TEST(execute, test_dot_scalar_scalar)
{
    auto shape = Shape{};
    auto A     = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B     = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt    = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f     = make_shared<Function>(make_shared<op::Dot>(A,B), rt, op::Parameters{A,B});

    auto external = make_shared<ngraph::runtime::ExternalFunction>(f);
    auto cf       = external->make_call_frame();

    // Create some tensors for input/output
    auto a      = ngraph::runtime::make_tensor<element::Float32>(shape);
    *a          = vector<float>{8};
    auto b      = ngraph::runtime::make_tensor<element::Float32>(shape);
    *b          = vector<float>{6};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape);

    (*cf)({a,b}, {result});
    ASSERT_EQ((vector<float>{48}), result->get_vector());
}

TEST(execute, test_dot_matrix_vector)
{
    auto shape_a = Shape{4,4};
    auto shape_b = Shape{4};
    auto A       = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto B       = make_shared<op::Parameter>(element::Float32::element_type(), shape_b);
    auto rt      = make_shared<TensorViewType>(element::Float32::element_type(), shape_b);
    auto f       = make_shared<Function>(make_shared<op::Dot>(A,B), rt, op::Parameters{A,B});
    auto shape_r = Shape{4};

    auto external = make_shared<ngraph::runtime::ExternalFunction>(f);
    auto cf       = external->make_call_frame();

    // Create some tensors for input/output
    auto a      = ngraph::runtime::make_tensor<element::Float32>(shape_a);
    *a          = vector<float>{ 1, 2, 3, 4,
                                 5, 6, 7, 8,
                                 9,10,11,12,
                                13,14,15,16};
    auto b      = ngraph::runtime::make_tensor<element::Float32>(shape_b);
    *b          = vector<float>{17,18,19,20};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape_r);

    (*cf)({a,b}, {result});
    ASSERT_EQ((vector<float>{190,486,782,1078}), result->get_vector());
}

TEST(execute, test_lessthan)
{
    auto shape = Shape{2, 2, 2};
    auto A     = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B     = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt    = make_shared<TensorViewType>(element::Bool::element_type(), shape);
    auto f     = make_shared<Function>(make_shared<op::Less>(A, B), rt, op::Parameters{A, B});

    auto external = make_shared<ngraph::runtime::ExternalFunction>(f);
    auto cf       = external->make_call_frame();

    // Create some tensors for input/output
    auto a      = ngraph::runtime::make_tensor<element::Float32>(shape);
    *a          = vector<float>{1, 8, -8, 17, -0.5, 0.5, 2, 1};
    auto b      = ngraph::runtime::make_tensor<element::Float32>(shape);
    *b          = vector<float>{1, 2, 4, 8, 0, 0, 1, 1.5};
    auto result = ngraph::runtime::make_tensor<element::Bool>(shape);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<char>{0, 0, 1, 0, 1, 0, 0, 1}), result->get_vector());
}

TEST(execute, test_log)
{
    auto shape = Shape{2, 2, 2};
    auto A     = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt    = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f     = make_shared<Function>(make_shared<op::Log>(A), rt, op::Parameters{A});

    auto external = make_shared<ngraph::runtime::ExternalFunction>(f);
    auto cf       = external->make_call_frame();

    // Create some tensors for input/output
    auto a = ngraph::runtime::make_tensor<element::Float32>(shape);
    *a     = vector<float>{expf(1), expf(2), expf(3), expf(4), expf(5), expf(6), expf(7), expf(8)};
    vector<float> loga;
    for (auto elt : a->get_vector()){
        loga.push_back(logf(elt));
    }
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape);

    (*cf)({a}, {result});
    ASSERT_EQ(loga, result->get_vector());
}

TEST(execute, test_maximum)
{
    auto shape = Shape{2, 2, 2};
    auto A     = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B     = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt    = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f     = make_shared<Function>(make_shared<op::Maximum>(A, B), rt, op::Parameters{A, B});

    auto external = make_shared<ngraph::runtime::ExternalFunction>(f);
    auto cf       = external->make_call_frame();

    // Create some tensors for input/output
    auto a      = ngraph::runtime::make_tensor<element::Float32>(shape);
    *a          = vector<float>{1, 8, -8, 17, -0.5, 0.5, 2, 1};
    auto b      = ngraph::runtime::make_tensor<element::Float32>(shape);
    *b          = vector<float>{1, 2, 4, 8, 0, 0, 1, 1.5};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<float>{1, 8, 4, 17, 0, 0.5, 2, 1.5}), result->get_vector());
}

TEST(execute, test_negative)
{
    auto shape = Shape{2, 3};
    auto A     = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt    = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f     = make_shared<Function>(make_shared<op::Negative>(A), rt, op::Parameters{A});

    auto external = make_shared<ngraph::runtime::ExternalFunction>(f);
    auto cf       = external->make_call_frame();

    // Create some tensors for input/output
    auto a      = ngraph::runtime::make_tensor<element::Float32>(shape);
    *a          = vector<float>{1, -2, 0, -4.8f, 8.6f, -8.6f};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape);

    (*cf)({a}, {result});
    ASSERT_EQ((vector<float>{-1, 2, 0, 4.8f, -8.6f, 8.6f}), result->get_vector());
}

TEST(execute, test_notequal)
{
    auto shape = Shape{2, 2, 2};
    auto A     = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B     = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt    = make_shared<TensorViewType>(element::Bool::element_type(), shape);
    auto f     = make_shared<Function>(make_shared<op::NotEqual>(A, B), rt, op::Parameters{A, B});

    auto external = make_shared<ngraph::runtime::ExternalFunction>(f);
    auto cf       = external->make_call_frame();

    // Create some tensors for input/output
    auto a      = ngraph::runtime::make_tensor<element::Float32>(shape);
    *a          = vector<float>{1, 8, -8, 17, -0.5, 0, 1, 1};
    auto b      = ngraph::runtime::make_tensor<element::Float32>(shape);
    *b          = vector<float>{1, 8, 4, 8, 0, 0, 1, 1.5};
    auto result = ngraph::runtime::make_tensor<element::Bool>(shape);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<char>{0, 0, 1, 1, 1, 0, 0, 1}), result->get_vector());
}

TEST(execute, test_select)
{
    auto shape = Shape{2, 2, 2};
    auto A     = make_shared<op::Parameter>(element::Bool::element_type(), shape);
    auto B     = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto C     = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt    = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f     = make_shared<Function>(make_shared<op::Select>(A, B, C), rt, op::Parameters{A, B, C});

    auto external = make_shared<ngraph::runtime::ExternalFunction>(f);
    auto cf       = external->make_call_frame();

    // Create some tensors for input/output
    auto a      = ngraph::runtime::make_tensor<element::Bool>(shape);
    *a          = vector<char>{0, 1, 1, 0, 0, 1, 0, 1};
    auto b      = ngraph::runtime::make_tensor<element::Float32>(shape);
    *b          = vector<float>{1, 2, 3, 4, 5, 6, 7, 8};
    auto c      = ngraph::runtime::make_tensor<element::Float32>(shape);
    *c          = vector<float>{11, 12, 13, 14, 15, 16, 17, 18};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape);

    (*cf)({a, b, c}, {result});
    ASSERT_EQ((vector<float>{11, 2, 3, 14, 15, 6, 17, 8}), result->get_vector());
}

TEST(execute, test_subtract)
{
    auto shape = Shape{2, 2};
    auto A     = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B     = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt    = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f     = make_shared<Function>(make_shared<op::Subtract>(A, B), rt, op::Parameters{A, B});

    auto external = make_shared<ngraph::runtime::ExternalFunction>(f);
    auto cf       = external->make_call_frame();

    // Create some tensors for input/output
    auto a      = ngraph::runtime::make_tensor<element::Float32>(shape);
    *a          = vector<float>{2, 4, 8, 16};
    auto b      = ngraph::runtime::make_tensor<element::Float32>(shape);
    *b          = vector<float>{1, 2, 4, 8};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape);

    (*cf)({a, b}, {result});
    ASSERT_EQ((vector<float>{1, 2, 4, 8}), result->get_vector());
}

TEST(execute, test_scalar_constant)
{
    auto shape = Shape{};
    auto A     = make_shared<op::ScalarConstant<element::Float32>>(-3.0f);
    auto rt    = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f     = make_shared<Function>(A, rt, op::Parameters{});

    auto external = make_shared<ngraph::runtime::ExternalFunction>(f);
    auto cf       = external->make_call_frame();

    // Create some tensors for input/output
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape);

    (*cf)({}, {result});
    ASSERT_EQ((vector<float>{-3.0f}), result->get_vector());
}

TEST(execute, test_tensor_constant)
{
    auto shape = Shape{2,2,2};
    auto A     = make_shared<op::TensorConstant<element::Float32>>(shape);
    A->get_value()->get_vector() = {1,2,3,4,5,6,7,8};
    auto rt    = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f     = make_shared<Function>(A, rt, op::Parameters{});

    auto external = make_shared<ngraph::runtime::ExternalFunction>(f);
    auto cf       = external->make_call_frame();

    // Create some tensors for input/output
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape);

    (*cf)({}, {result});
    ASSERT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8}), result->get_vector());
}

TEST(execute, test_tensor_constant_with_op)
{
    auto shape = Shape{2,2,2};
    auto A     = make_shared<op::TensorConstant<element::Float32>>(shape);
    A->get_value()->get_vector() = {-1,2,3,-4,5,-6,-7,8};
    auto rt    = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f     = make_shared<Function>(make_shared<op::Abs>(A), rt, op::Parameters{});

    auto external = make_shared<ngraph::runtime::ExternalFunction>(f);
    auto cf       = external->make_call_frame();

    // Create some tensors for input/output
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape);

    (*cf)({}, {result});
    ASSERT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8}), result->get_vector());
}

TEST(execute, test_function_call)
{
    // First create "f(A,B,C) = (A+B)*C".
    auto shape = Shape{2, 2};
    auto A     = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B     = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto C     = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt_f  = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f     = make_shared<Function>((A + B) * C, rt_f, op::Parameters{A, B, C});

    // Now make "g(X,Y,Z) = f(X,Y,Z) + f(X,Y,Z)"
    auto X     = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto Y     = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto Z     = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt_g  = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto g     = make_shared<Function>(
                     make_shared<op::FunctionCall>(f,Nodes{X,Y,Z})
                     + make_shared<op::FunctionCall>(f,Nodes{X,Y,Z}),
                     rt_g,
                     op::Parameters{X, Y, Z});

    // Now call g on some test vectors.
    auto external = make_shared<ngraph::runtime::ExternalFunction>(g);
    auto cf       = external->make_call_frame();

    auto x      = ngraph::runtime::make_tensor<element::Float32>(shape);
    *x          = vector<float>{1, 2, 3, 4};
    auto y      = ngraph::runtime::make_tensor<element::Float32>(shape);
    *y          = vector<float>{5, 6, 7, 8};
    auto z      = ngraph::runtime::make_tensor<element::Float32>(shape);
    *z          = vector<float>{9, 10, 11, 12};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape);

    (*cf)({x, y, z}, {result});
    ASSERT_EQ((vector<float>{108, 160, 220, 288}), result->get_vector());

    (*cf)({y, x, z}, {result});
    ASSERT_EQ((vector<float>{108, 160, 220, 288}), result->get_vector());

    (*cf)({x, z, y}, {result});
    ASSERT_EQ((vector<float>{100, 144, 196, 256}), result->get_vector());
}

TEST(execute, test_broadcast_scalar_vector)
{
    auto shape_a = Shape{};
    auto A       = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_r = Shape{4};
    auto rt      = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto f       = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{0}), rt, op::Parameters{A});

    auto external = make_shared<ngraph::runtime::ExternalFunction>(f);
    auto cf       = external->make_call_frame();

    // Create some tensors for input/output
    auto a      = ngraph::runtime::make_tensor<element::Float32>(shape_a);
    *a          = vector<float>{6};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape_r);

    (*cf)({a}, {result});
    ASSERT_EQ((vector<float>{6, 6, 6, 6}), result->get_vector());
}

TEST(execute, test_broadcast_scalar_matrix)
{
    auto shape_a = Shape{};
    auto A       = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_r = Shape{2,2};
    auto rt      = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto f       = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{0,1}), rt, op::Parameters{A});

    auto external = make_shared<ngraph::runtime::ExternalFunction>(f);
    auto cf       = external->make_call_frame();

    // Create some tensors for input/output
    auto a      = ngraph::runtime::make_tensor<element::Float32>(shape_a);
    *a          = vector<float>{6};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape_r);

    (*cf)({a}, {result});
    ASSERT_EQ((vector<float>{6, 6, 6, 6}), result->get_vector());
}

TEST(execute, test_broadcast_scalar_tensor)
{
    auto shape_a = Shape{};
    auto A       = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_r = Shape{2,2,2};
    auto rt      = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto f       = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{0,1,2}), rt, op::Parameters{A});

    auto external = make_shared<ngraph::runtime::ExternalFunction>(f);
    auto cf       = external->make_call_frame();

    // Create some tensors for input/output
    auto a      = ngraph::runtime::make_tensor<element::Float32>(shape_a);
    *a          = vector<float>{6};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape_r);

    (*cf)({a}, {result});
    ASSERT_EQ((vector<float>{6, 6, 6, 6, 6, 6, 6, 6}), result->get_vector());
}

TEST(execute, test_broadcast_trivial)
{
    auto shape = Shape{2,2,2};
    auto A     = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto rt    = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f     = make_shared<Function>(make_shared<op::Broadcast>(A, shape, AxisSet{}), rt, op::Parameters{A});

    auto external = make_shared<ngraph::runtime::ExternalFunction>(f);
    auto cf       = external->make_call_frame();

    // Create some tensors for input/output
    auto a      = ngraph::runtime::make_tensor<element::Float32>(shape);
    *a          = vector<float>{2, 4, 6, 8, 16, 32, 64, 128};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape);

    (*cf)({a}, {result});
    ASSERT_EQ((vector<float>{2, 4, 6, 8, 16, 32, 64, 128}), result->get_vector());
}

TEST(execute, test_broadcast_vector_colwise)
{
    auto shape_a = Shape{3};
    auto A       = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_r = Shape{3,4};
    auto rt      = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto f       = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{1}), rt, op::Parameters{A});

    auto external = make_shared<ngraph::runtime::ExternalFunction>(f);
    auto cf       = external->make_call_frame();

    // Create some tensors for input/output
    auto a      = ngraph::runtime::make_tensor<element::Float32>(shape_a);
    *a          = vector<float>{1,2,3};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape_r);

    (*cf)({a}, {result});
    ASSERT_EQ((vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}), result->get_vector());
}

TEST(execute, test_broadcast_vector_rowwise)
{
    auto shape_a = Shape{4};
    auto A       = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_r = Shape{3,4};
    auto rt      = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto f       = make_shared<Function>(make_shared<op::Broadcast>(A, shape_r, AxisSet{0}), rt, op::Parameters{A});

    auto external = make_shared<ngraph::runtime::ExternalFunction>(f);
    auto cf       = external->make_call_frame();

    // Create some tensors for input/output
    auto a      = ngraph::runtime::make_tensor<element::Float32>(shape_a);
    *a          = vector<float>{1,2,3,4};
    auto result = ngraph::runtime::make_tensor<element::Float32>(shape_r);

    (*cf)({a}, {result});
    ASSERT_EQ((vector<float>{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}), result->get_vector());
}
