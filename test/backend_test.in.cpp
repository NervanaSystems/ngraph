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
#include "ngraph/serializer.hpp"
#include "util/all_close.hpp"
#include "util/ndarray.hpp"

using namespace std;
using namespace ngraph;

template <typename T>
static void copy_data(shared_ptr<runtime::TensorView> tv, const vector<T>& data)
{
    size_t data_size = data.size() * sizeof(T);
    tv->write(data.data(), 0, data_size);
}

TEST(${BACKEND_NAME}, parameter_as_output)
{
    auto shape = Shape{3, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto rt = make_shared<TensorViewType>(element::f32, shape);
    auto f = make_shared<Function>(A, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    shared_ptr<runtime::TensorView> a = backend->make_primary_tensor_view(element::f32, shape);
    shared_ptr<runtime::TensorView> result = backend->make_primary_tensor_view(element::f32, shape);

    vector<float> expected{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    vector<float> zero(shape_size(shape), 0);
    copy_data(a, expected);

    cf->call({a}, {result});
    EXPECT_EQ(result->get_vector<float>(), expected);
}

TEST(${BACKEND_NAME}, ab)
{
    using f32 = element::Float32;

    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(f32::element_type(), shape);
    auto B = make_shared<op::Parameter>(f32::element_type(), shape);
    auto rt = make_shared<TensorViewType>(f32::element_type(), shape);
    auto f = make_shared<Function>(A + B, rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    shared_ptr<runtime::TensorView> a =
        backend->make_primary_tensor_view(f32::element_type(), shape);
    shared_ptr<runtime::TensorView> b =
        backend->make_primary_tensor_view(f32::element_type(), shape);
    shared_ptr<runtime::TensorView> result =
        backend->make_primary_tensor_view(f32::element_type(), shape);

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());

    cf->call({a, b}, {result});
    EXPECT_EQ(result->get_vector<float>(),
              (test::NDArray<float, 2>({{6, 8}, {10, 12}})).get_vector());
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

    copy_data(a, test::NDArray<float, 2>({{1, 2}, {3, 4}}).get_vector());
    copy_data(b, test::NDArray<float, 2>({{5, 6}, {7, 8}}).get_vector());
    copy_data(c, test::NDArray<float, 2>({{9, 10}, {11, 12}}).get_vector());

    cf->call({a, b, c}, {result});
    EXPECT_EQ(result->get_vector<float>(),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

    cf->call({b, a, c}, {result});
    EXPECT_EQ(result->get_vector<float>(),
              (test::NDArray<float, 2>({{54, 80}, {110, 144}})).get_vector());

    cf->call({a, c, b}, {result});
    EXPECT_EQ(result->get_vector<float>(),
              (test::NDArray<float, 2>({{50, 72}, {98, 128}})).get_vector());
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
TEST(${BACKEND_NAME}, tuple_abc)
{
    auto shape = Shape{2, 2};

    auto tensor_view_type = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto rt = make_shared<TupleType>(ValueTypes{tensor_view_type});

    auto ABC = make_shared<op::Parameter>(
        make_shared<TupleType>(ValueTypes{tensor_view_type, tensor_view_type, tensor_view_type}));

    auto A = make_shared<op::XLAGetTupleElement>(ABC, 0);
    auto B = make_shared<op::XLAGetTupleElement>(ABC, 1);
    auto C = make_shared<op::XLAGetTupleElement>(ABC, 2);
    auto f = make_shared<XLAFunction>(
        make_shared<op::XLATuple>(Nodes{(A + B) * C}), rt, op::Parameters{ABC});

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
    EXPECT_EQ((vector<float>{54, 80, 110, 144}), result->get_vector<float>());

    cf->call({bac}, {result_tuple});
    EXPECT_EQ((vector<float>{54, 80, 110, 144}), result->get_vector<float>());

    cf->call({acb}, {result_tuple});
    EXPECT_EQ((vector<float>{50, 72, 98, 128}), result->get_vector<float>());
}

// Same as abc, but using tuples for input and output
TEST(${BACKEND_NAME}, tuple_abc_int64)
{
    auto shape = Shape{2, 2};

    auto tensor_view_type = make_shared<TensorViewType>(element::Int64::element_type(), shape);
    auto rt = make_shared<TupleType>(ValueTypes{tensor_view_type});

    auto ABC = make_shared<op::Parameter>(
        make_shared<TupleType>(ValueTypes{tensor_view_type, tensor_view_type, tensor_view_type}));

    auto A = make_shared<op::XLAGetTupleElement>(ABC, 0);
    auto B = make_shared<op::XLAGetTupleElement>(ABC, 1);
    auto C = make_shared<op::XLAGetTupleElement>(ABC, 2);
    auto f = make_shared<XLAFunction>(
        make_shared<op::XLATuple>(Nodes{(A + B) * C}), rt, op::Parameters{ABC});

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
    EXPECT_EQ((vector<element::Int64::type>{54, 80, 110, 144}),
              result->get_vector<element::Int64::type>());

    cf->call({bac}, {result_tuple});
    EXPECT_EQ((vector<element::Int64::type>{54, 80, 110, 144}),
              result->get_vector<element::Int64::type>());

    cf->call({acb}, {result_tuple});
    EXPECT_EQ((vector<element::Int64::type>{50, 72, 98, 128}),
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
    auto f = make_shared<XLAFunction>(
        make_shared<op::XLATuple>(Nodes{A_add_B, A_add_B_mul_C}), rt, op::Parameters{A, B, C});

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

    EXPECT_EQ((vector<float>{6, 8, 10, 12}), r0->get_vector<float>());
    EXPECT_EQ((vector<float>{54, 80, 110, 144}), r1->get_vector<float>());
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
    EXPECT_EQ((vector<float>{1, 2, 0, 4.8f}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, ceiling)
{
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto result_type = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Ceiling>(A), result_type, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{-2.5f, -2.0f, 0.3f, 4.8f});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    cf->call({a}, {result});
    EXPECT_EQ((vector<float>{-2.0f, -2.0f, 1.0f, 5.0f}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{2, 4, 1, 2, 4, 2, 3, 5, 8, 16, 8, 16, 32, 7, 11, 13}),
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
    EXPECT_EQ((vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13}),
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
    EXPECT_EQ((vector<element::Int64::type>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13}),
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
    EXPECT_EQ((vector<float>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 18, 19}),
              result->get_vector<float>());
}

// from numpy import *
// a=linspace(1,2*3*4*3*2,2*3*4*3*2)
// b=linspace(1000+1,1000+2*3*3*3*2,2*3*3*3*2)
// c=linspace(2000+1,2000+2*3*2*3*2,2*3*2*3*2)
// a.shape=(2,3,4,3,2)
// b.shape=(2,3,3,3,2)
// c.shape=(2,3,2,3,2)
// z=concatenate((a,b,c),axis=2)
// z.shape=(2*3*(4+3+2)*3*2)
// set_printoptions(suppress=True)
// print(z)
//
// [    1.     2.     3.     4.     5.     6.     7.     8.     9.    10.
//     11.    12.    13.    14.    15.    16.    17.    18.    19.    20.
//     21.    22.    23.    24.  1001.  1002.  1003.  1004.  1005.  1006.
//   1007.  1008.  1009.  1010.  1011.  1012.  1013.  1014.  1015.  1016.
//   1017.  1018.  2001.  2002.  2003.  2004.  2005.  2006.  2007.  2008.
//   2009.  2010.  2011.  2012.    25.    26.    27.    28.    29.    30.
//     31.    32.    33.    34.    35.    36.    37.    38.    39.    40.
//     41.    42.    43.    44.    45.    46.    47.    48.  1019.  1020.
//   1021.  1022.  1023.  1024.  1025.  1026.  1027.  1028.  1029.  1030.
//   1031.  1032.  1033.  1034.  1035.  1036.  2013.  2014.  2015.  2016.
//   2017.  2018.  2019.  2020.  2021.  2022.  2023.  2024.    49.    50.
//     51.    52.    53.    54.    55.    56.    57.    58.    59.    60.
//     61.    62.    63.    64.    65.    66.    67.    68.    69.    70.
//     71.    72.  1037.  1038.  1039.  1040.  1041.  1042.  1043.  1044.
//   1045.  1046.  1047.  1048.  1049.  1050.  1051.  1052.  1053.  1054.
//   2025.  2026.  2027.  2028.  2029.  2030.  2031.  2032.  2033.  2034.
//   2035.  2036.    73.    74.    75.    76.    77.    78.    79.    80.
//     81.    82.    83.    84.    85.    86.    87.    88.    89.    90.
//     91.    92.    93.    94.    95.    96.  1055.  1056.  1057.  1058.
//   1059.  1060.  1061.  1062.  1063.  1064.  1065.  1066.  1067.  1068.
//   1069.  1070.  1071.  1072.  2037.  2038.  2039.  2040.  2041.  2042.
//   2043.  2044.  2045.  2046.  2047.  2048.    97.    98.    99.   100.
//    101.   102.   103.   104.   105.   106.   107.   108.   109.   110.
//    111.   112.   113.   114.   115.   116.   117.   118.   119.   120.
//   1073.  1074.  1075.  1076.  1077.  1078.  1079.  1080.  1081.  1082.
//   1083.  1084.  1085.  1086.  1087.  1088.  1089.  1090.  2049.  2050.
//   2051.  2052.  2053.  2054.  2055.  2056.  2057.  2058.  2059.  2060.
//    121.   122.   123.   124.   125.   126.   127.   128.   129.   130.
//    131.   132.   133.   134.   135.   136.   137.   138.   139.   140.
//    141.   142.   143.   144.  1091.  1092.  1093.  1094.  1095.  1096.
//   1097.  1098.  1099.  1100.  1101.  1102.  1103.  1104.  1105.  1106.
//   1107.  1108.  2061.  2062.  2063.  2064.  2065.  2066.  2067.  2068.
//   2069.  2070.  2071.  2072.]
TEST(${BACKEND_NAME}, concat_5d)
{
    vector<float> a_data(2 * 3 * 4 * 3 * 2);
    for (int i = 0; i < 2 * 3 * 4 * 3 * 2; i++)
    {
        a_data[i] = float(i + 1);
    }

    vector<float> b_data(2 * 3 * 3 * 3 * 2);
    for (int i = 0; i < 2 * 3 * 3 * 3 * 2; i++)
    {
        b_data[i] = 1000 + float(i + 1);
    }

    vector<float> c_data(2 * 3 * 2 * 3 * 2);
    for (int i = 0; i < 2 * 3 * 2 * 3 * 2; i++)
    {
        c_data[i] = 2000 + float(i + 1);
    }

    auto shape_a = Shape{2, 3, 4, 3, 2};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_b = Shape{2, 3, 3, 3, 2};
    auto B = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_b));
    auto shape_c = Shape{2, 3, 2, 3, 2};
    auto C = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_c));
    auto shape_r = Shape{2, 3, 9, 3, 2};

    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Concat>(Nodes{A, B, C}, 2);
    auto f = make_shared<Function>(r, rt, op::Parameters{A, B, C});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, a_data);
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape_b);
    copy_data(b, b_data);
    auto c = backend->make_primary_tensor_view(element::Float32::element_type(), shape_c);
    copy_data(c, c_data);

    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a, b, c}, {result});
    EXPECT_EQ(
        (vector<float>{
            1.,    2.,    3.,    4.,    5.,    6.,    7.,    8.,    9.,    10.,   11.,   12.,
            13.,   14.,   15.,   16.,   17.,   18.,   19.,   20.,   21.,   22.,   23.,   24.,
            1001., 1002., 1003., 1004., 1005., 1006., 1007., 1008., 1009., 1010., 1011., 1012.,
            1013., 1014., 1015., 1016., 1017., 1018., 2001., 2002., 2003., 2004., 2005., 2006.,
            2007., 2008., 2009., 2010., 2011., 2012., 25.,   26.,   27.,   28.,   29.,   30.,
            31.,   32.,   33.,   34.,   35.,   36.,   37.,   38.,   39.,   40.,   41.,   42.,
            43.,   44.,   45.,   46.,   47.,   48.,   1019., 1020., 1021., 1022., 1023., 1024.,
            1025., 1026., 1027., 1028., 1029., 1030., 1031., 1032., 1033., 1034., 1035., 1036.,
            2013., 2014., 2015., 2016., 2017., 2018., 2019., 2020., 2021., 2022., 2023., 2024.,
            49.,   50.,   51.,   52.,   53.,   54.,   55.,   56.,   57.,   58.,   59.,   60.,
            61.,   62.,   63.,   64.,   65.,   66.,   67.,   68.,   69.,   70.,   71.,   72.,
            1037., 1038., 1039., 1040., 1041., 1042., 1043., 1044., 1045., 1046., 1047., 1048.,
            1049., 1050., 1051., 1052., 1053., 1054., 2025., 2026., 2027., 2028., 2029., 2030.,
            2031., 2032., 2033., 2034., 2035., 2036., 73.,   74.,   75.,   76.,   77.,   78.,
            79.,   80.,   81.,   82.,   83.,   84.,   85.,   86.,   87.,   88.,   89.,   90.,
            91.,   92.,   93.,   94.,   95.,   96.,   1055., 1056., 1057., 1058., 1059., 1060.,
            1061., 1062., 1063., 1064., 1065., 1066., 1067., 1068., 1069., 1070., 1071., 1072.,
            2037., 2038., 2039., 2040., 2041., 2042., 2043., 2044., 2045., 2046., 2047., 2048.,
            97.,   98.,   99.,   100.,  101.,  102.,  103.,  104.,  105.,  106.,  107.,  108.,
            109.,  110.,  111.,  112.,  113.,  114.,  115.,  116.,  117.,  118.,  119.,  120.,
            1073., 1074., 1075., 1076., 1077., 1078., 1079., 1080., 1081., 1082., 1083., 1084.,
            1085., 1086., 1087., 1088., 1089., 1090., 2049., 2050., 2051., 2052., 2053., 2054.,
            2055., 2056., 2057., 2058., 2059., 2060., 121.,  122.,  123.,  124.,  125.,  126.,
            127.,  128.,  129.,  130.,  131.,  132.,  133.,  134.,  135.,  136.,  137.,  138.,
            139.,  140.,  141.,  142.,  143.,  144.,  1091., 1092., 1093., 1094., 1095., 1096.,
            1097., 1098., 1099., 1100., 1101., 1102., 1103., 1104., 1105., 1106., 1107., 1108.,
            2061., 2062., 2063., 2064., 2065., 2066., 2067., 2068., 2069., 2070., 2071., 2072.}),
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
    EXPECT_EQ((vector<float>{2, 2, 2, 2}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{std::numeric_limits<float>::infinity(),
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
    EXPECT_EQ((vector<char>{1, 1, 0, 0, 0, 1, 1, 0}), result->get_vector<char>());
}

TEST(${BACKEND_NAME}, floor)
{
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto result_type = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Floor>(A), result_type, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{-2.5f, -2.0f, 0.3f, 4.8f});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    cf->call({a}, {result});
    EXPECT_EQ((vector<float>{-3.0f, -2.0f, 0.0f, 4.0f}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{0}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{0, 0, 0, 0}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, dot_matrix_3x2_2x0)
{
    auto shape_a = Shape{3, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape_a);
    auto shape_b = Shape{2, 0};
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape_b);
    auto shape_r = Shape{3, 0};
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
    EXPECT_EQ((vector<float>{}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{0, 0}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, dot1d)
{
    auto shape = Shape{4};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto shape_r = Shape{};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
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
    EXPECT_EQ((vector<float>{170}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{19, 22, 43, 50}), result->get_vector<float>());
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
// >>> tensordot(a,b,axes=([2],[0]))
// array([[[[ 11.,  14.],
//          [ 17.,  20.]],
//
//         [[ 23.,  30.],
//          [ 37.,  44.]]],
//
//
//        [[[ 35.,  46.],
//          [ 57.,  68.]],
//
//         [[ 47.,  62.],
//          [ 77.,  92.]]]])
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
    EXPECT_EQ((vector<float>{11, 14, 17, 20, 23, 30, 37, 44, 35, 46, 57, 68, 47, 62, 77, 92}),
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
    EXPECT_EQ((vector<float>{20,  23,  26,  29,  56,  68,  80,  92,  92,  113, 134,
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
    EXPECT_EQ((vector<float>{6, 12, 18, 24, 30, 36, 42, 48}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{6, 12, 18, 24, 30, 36, 42, 48}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{48}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{190, 486, 782, 1078}), result->get_vector<float>());
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
    EXPECT_EQ((vector<element::Int64::type>{190, 486, 782, 1078}),
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
    EXPECT_EQ((vector<char>{0, 1, 0, 1, 0, 1, 1, 0}), result->get_vector<char>());
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
    EXPECT_EQ((vector<char>{1, 1, 1, 1, 0, 1, 1, 0}), result->get_vector<char>());
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
    EXPECT_EQ((vector<char>{0, 0, 1, 0, 1, 0, 0, 1}), result->get_vector<char>());
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
    EXPECT_EQ((vector<char>{1, 0, 1, 0, 1, 1, 0, 1}), result->get_vector<char>());
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
    EXPECT_EQ((vector<char>{0, 0, 0, 0, 0, 0, 0, 0}), result->get_vector<char>());
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
    EXPECT_EQ(loga, result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{1, 8, 4, 17, 0, 0.5, 2, 1.5}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{1, 2, -8, 8, -.5, 0, 1, 1}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{-1, 2, 0, 4.8f, -8.6f, 8.6f}), result->get_vector<float>());
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
    EXPECT_EQ((vector<char>{0, 0, 1, 1, 1, 0, 0, 1}), result->get_vector<char>());
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
    EXPECT_EQ((vector<float>{11, 2, 3, 14, 15, 6, 17, 8}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{1, 2, 4, 8}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, tensor_constant)
{
    auto shape = Shape{2, 2, 2};
    auto A = op::Constant::create(element::f32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(A, rt, op::Parameters{});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    cf->call({}, {result});
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, tensor_constant_with_op)
{
    auto shape = Shape{2, 2, 2};
    auto A = op::Constant::create(element::f32, shape, {-1, 2, 3, -4, 5, -6, -7, 8});
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Abs>(A), rt, op::Parameters{});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    cf->call({}, {result});
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, constant_broadcast)
{
    const string js =
        R"([{
       "name" : "Function_0",
       "ops" : [
           {
             "element_type" :
                 {"bitwidth" : 32, "c_type_string" : "float", "is_real" : true, "is_signed" : true},
             "inputs" : [],
             "name" : "Parameter_4",
             "op" : "Parameter",
             "outputs" : ["Parameter_4"],
             "shape" : [ 3, 4 ]
           },
           {
             "element_type" :
                 {"bitwidth" : 32, "c_type_string" : "float", "is_real" : true, "is_signed" : true},
             "inputs" : [],
             "name" : "Parameter_0",
             "op" : "Parameter",
             "outputs" : ["Parameter_0"],
             "shape" : [ 3, 4 ]
           },
           {
             "element_type" :
                 {"bitwidth" : 32, "c_type_string" : "float", "is_real" : true, "is_signed" : true},
             "inputs" : [],
             "name" : "Constant_1",
             "op" : "Constant",
             "outputs" : ["Constant_1"],
             "shape" : [],
             "value" : ["0"]
           },
           {
             "axes" : [ 0, 1 ],
             "element_type" :
                 {"bitwidth" : 32, "c_type_string" : "float", "is_real" : true, "is_signed" : true},
             "inputs" : ["Constant_1"],
             "name" : "Broadcast_2",
             "op" : "Broadcast",
             "outputs" : ["Broadcast_2"],
             "shape" : [ 3, 4 ]
           },
           {
             "element_type" :
                 {"bitwidth" : 32, "c_type_string" : "float", "is_real" : true, "is_signed" : true},
             "inputs" : [ "Parameter_0", "Broadcast_2" ],
             "name" : "Maximum_3",
             "op" : "Maximum",
             "outputs" : ["Maximum_3"]
           },
           {
             "element_type" :
                 {"bitwidth" : 32, "c_type_string" : "float", "is_real" : true, "is_signed" : true},
             "inputs" : [ "Maximum_3", "Parameter_4" ],
             "name" : "Multiply_5",
             "op" : "Multiply",
             "outputs" : ["Multiply_5"]
           }
       ],
       "parameters" : [ "Parameter_0", "Parameter_4" ],
       "result" : ["Multiply_5"],
       "result_shape" : [ 3, 4 ],
       "result_type" :
           {"bitwidth" : 32, "c_type_string" : "float", "is_real" : true, "is_signed" : true}
    }])";
    stringstream ss(js);

    shared_ptr<Function> f = ngraph::deserialize(ss);

    // max(x,broadcast(Constant(0)))
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // If this compiles it works
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
    EXPECT_EQ((vector<float>{108, 160, 220, 288}), result->get_vector<float>());

    cf->call({y, x, z}, {result});
    EXPECT_EQ((vector<float>{108, 160, 220, 288}), result->get_vector<float>());

    cf->call({x, z, y}, {result});
    EXPECT_EQ((vector<float>{100, 144, 196, 256}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{6, 6, 6, 6}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{6, 6, 6, 6}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{6, 6, 6, 6, 6, 6, 6, 6}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{2, 4, 6, 8, 16, 32, 64, 128}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}), result->get_vector<float>());
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
    EXPECT_EQ((vector<element::Int64::type>{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}),
              result->get_vector<element::Int64::type>());
}

TEST(${BACKEND_NAME}, broadcast_matrix_0)
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
    EXPECT_EQ((vector<element::Float32::type>{1, 2, 3, 4, 1, 2, 3, 4}),
              result->get_vector<element::Float32::type>());
}

TEST(${BACKEND_NAME}, broadcast_matrix_1)
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
    EXPECT_EQ((vector<element::Float32::type>{1, 2, 1, 2, 3, 4, 3, 4}),
              result->get_vector<element::Float32::type>());
}

TEST(${BACKEND_NAME}, broadcast_matrix_2)
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
    EXPECT_EQ((vector<element::Float32::type>{1, 1, 2, 2, 3, 3, 4, 4}),
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
    EXPECT_EQ((vector<element::Float32::type>{1, 2, 3, 4}), result->get_vector<float>());
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
    EXPECT_EQ((vector<element::Bool::type>{1, 2, 3, 4}), result->get_vector<element::Bool::type>());
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
    EXPECT_EQ((vector<element::Bool::type>{1, 2, 3, 4}), result->get_vector<element::Bool::type>());
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
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{10}), result->get_vector<float>());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), a->get_vector<float>());
    EXPECT_EQ((vector<float>{0}), b->get_vector<float>());
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
    EXPECT_EQ((vector<float>{9, 12}), result->get_vector<float>());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), a->get_vector<float>());
    EXPECT_EQ((vector<float>{0}), b->get_vector<float>());
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
    EXPECT_EQ((vector<float>{3, 7, 11}), result->get_vector<float>());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), a->get_vector<float>());
    EXPECT_EQ((vector<float>{0}), b->get_vector<float>());
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
    EXPECT_EQ((vector<float>{66, 66, 66}), result->get_vector<float>());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), a->get_vector<float>());
    EXPECT_EQ((vector<float>{66}), b->get_vector<float>());
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
    EXPECT_EQ((vector<float>{77, 77}), result->get_vector<float>());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), a->get_vector<float>());
    EXPECT_EQ((vector<float>{77}), b->get_vector<float>());
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
    EXPECT_EQ((vector<float>{88}), result->get_vector<float>());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), a->get_vector<float>());
    EXPECT_EQ((vector<float>{88}), b->get_vector<float>());
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
    EXPECT_EQ((vector<float>{99}), result->get_vector<float>());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), a->get_vector<float>());
    EXPECT_EQ((vector<float>{99}), b->get_vector<float>());
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
    EXPECT_EQ((vector<float>{1.0f * 10.0f * 19.0f * 4.0f * 13.0f * 22.0f * 7.0f * 16.0f * 25.0f,
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
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{6}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{6}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{42}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{1, 2, 3}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{1, 2, 3}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{1, 2, 3}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{1, 4, 7, 2, 5, 8, 3, 6, 9}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{1, 3, 5, 2, 4, 6}), result->get_vector<float>());
}

//
// Numpy:
//
// >>> x = linspace(1,2*2*3*3*2*4,2*2*3*3*2*4)
// >>> x.shape=(2,2,3,3,2,4)
// >>> y = ascontiguousarray(transpose(x,(2,4,0,5,3,1)))
// >>> y.shape=2*2*3*3*2*4
// >>> y
// array([   1.,   73.,    9.,   81.,   17.,   89.,    2.,   74.,   10.,
//          82.,   18.,   90.,    3.,   75.,   11.,   83.,   19.,   91.,
//           4.,   76.,   12.,   84.,   20.,   92.,  145.,  217.,  153.,
//         225.,  161.,  233.,  146.,  218.,  154.,  226.,  162.,  234.,
//         147.,  219.,  155.,  227.,  163.,  235.,  148.,  220.,  156.,
//         228.,  164.,  236.,    5.,   77.,   13.,   85.,   21.,   93.,
//           6.,   78.,   14.,   86.,   22.,   94.,    7.,   79.,   15.,
//          87.,   23.,   95.,    8.,   80.,   16.,   88.,   24.,   96.,
//         149.,  221.,  157.,  229.,  165.,  237.,  150.,  222.,  158.,
//         230.,  166.,  238.,  151.,  223.,  159.,  231.,  167.,  239.,
//         152.,  224.,  160.,  232.,  168.,  240.,   25.,   97.,   33.,
//         105.,   41.,  113.,   26.,   98.,   34.,  106.,   42.,  114.,
//          27.,   99.,   35.,  107.,   43.,  115.,   28.,  100.,   36.,
//         108.,   44.,  116.,  169.,  241.,  177.,  249.,  185.,  257.,
//         170.,  242.,  178.,  250.,  186.,  258.,  171.,  243.,  179.,
//         251.,  187.,  259.,  172.,  244.,  180.,  252.,  188.,  260.,
//          29.,  101.,   37.,  109.,   45.,  117.,   30.,  102.,   38.,
//         110.,   46.,  118.,   31.,  103.,   39.,  111.,   47.,  119.,
//          32.,  104.,   40.,  112.,   48.,  120.,  173.,  245.,  181.,
//         253.,  189.,  261.,  174.,  246.,  182.,  254.,  190.,  262.,
//         175.,  247.,  183.,  255.,  191.,  263.,  176.,  248.,  184.,
//         256.,  192.,  264.,   49.,  121.,   57.,  129.,   65.,  137.,
//          50.,  122.,   58.,  130.,   66.,  138.,   51.,  123.,   59.,
//         131.,   67.,  139.,   52.,  124.,   60.,  132.,   68.,  140.,
//         193.,  265.,  201.,  273.,  209.,  281.,  194.,  266.,  202.,
//         274.,  210.,  282.,  195.,  267.,  203.,  275.,  211.,  283.,
//         196.,  268.,  204.,  276.,  212.,  284.,   53.,  125.,   61.,
//         133.,   69.,  141.,   54.,  126.,   62.,  134.,   70.,  142.,
//          55.,  127.,   63.,  135.,   71.,  143.,   56.,  128.,   64.,
//         136.,   72.,  144.,  197.,  269.,  205.,  277.,  213.,  285.,
//         198.,  270.,  206.,  278.,  214.,  286.,  199.,  271.,  207.,
//         279.,  215.,  287.,  200.,  272.,  208.,  280.,  216.,  288.])
//
// Disabled because it doesn't work on CPU yet.
//
TEST(DISABLED_${BACKEND_NAME}, reshape_6d)
{
    vector<float> a_data(2 * 2 * 3 * 3 * 2 * 4);
    for (int i = 0; i < 2 * 2 * 3 * 3 * 2 * 4; i++)
    {
        a_data[i] = float(i + 1);
    }

    auto shape_a = Shape{2, 2, 3, 3, 2, 4};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_r = Shape{3, 2, 2, 4, 3, 2};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);

    auto r = make_shared<op::Reshape>(A, AxisVector{2, 4, 0, 5, 3, 1}, shape_r);
    auto f = make_shared<Function>(r, rt, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, a_data);

    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a}, {result});
    EXPECT_EQ(
        (vector<float>{
            1.,   73.,  9.,   81.,  17.,  89.,  2.,   74.,  10.,  82.,  18.,  90.,  3.,   75.,
            11.,  83.,  19.,  91.,  4.,   76.,  12.,  84.,  20.,  92.,  145., 217., 153., 225.,
            161., 233., 146., 218., 154., 226., 162., 234., 147., 219., 155., 227., 163., 235.,
            148., 220., 156., 228., 164., 236., 5.,   77.,  13.,  85.,  21.,  93.,  6.,   78.,
            14.,  86.,  22.,  94.,  7.,   79.,  15.,  87.,  23.,  95.,  8.,   80.,  16.,  88.,
            24.,  96.,  149., 221., 157., 229., 165., 237., 150., 222., 158., 230., 166., 238.,
            151., 223., 159., 231., 167., 239., 152., 224., 160., 232., 168., 240., 25.,  97.,
            33.,  105., 41.,  113., 26.,  98.,  34.,  106., 42.,  114., 27.,  99.,  35.,  107.,
            43.,  115., 28.,  100., 36.,  108., 44.,  116., 169., 241., 177., 249., 185., 257.,
            170., 242., 178., 250., 186., 258., 171., 243., 179., 251., 187., 259., 172., 244.,
            180., 252., 188., 260., 29.,  101., 37.,  109., 45.,  117., 30.,  102., 38.,  110.,
            46.,  118., 31.,  103., 39.,  111., 47.,  119., 32.,  104., 40.,  112., 48.,  120.,
            173., 245., 181., 253., 189., 261., 174., 246., 182., 254., 190., 262., 175., 247.,
            183., 255., 191., 263., 176., 248., 184., 256., 192., 264., 49.,  121., 57.,  129.,
            65.,  137., 50.,  122., 58.,  130., 66.,  138., 51.,  123., 59.,  131., 67.,  139.,
            52.,  124., 60.,  132., 68.,  140., 193., 265., 201., 273., 209., 281., 194., 266.,
            202., 274., 210., 282., 195., 267., 203., 275., 211., 283., 196., 268., 204., 276.,
            212., 284., 53.,  125., 61.,  133., 69.,  141., 54.,  126., 62.,  134., 70.,  142.,
            55.,  127., 63.,  135., 71.,  143., 56.,  128., 64.,  136., 72.,  144., 197., 269.,
            205., 277., 213., 285., 198., 270., 206., 278., 214., 286., 199., 271., 207., 279.,
            215., 287., 200., 272., 208., 280., 216., 288.}),
        result->get_vector<float>());
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
    EXPECT_EQ(input, result->get_vector<float>());
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
    EXPECT_EQ(input, result->get_vector<float>());
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
    EXPECT_EQ(input, result->get_vector<float>());
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
    EXPECT_EQ(input, result->get_vector<float>());
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
    EXPECT_EQ(input, result->get_vector<float>());
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
    EXPECT_EQ(input, result->get_vector<float>());
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
    EXPECT_EQ(input, result->get_vector<float>());
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
    EXPECT_EQ(input, result->get_vector<float>());
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
    EXPECT_EQ(input, result->get_vector<float>());
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
    EXPECT_EQ(
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
    EXPECT_EQ((vector<float>{312}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{2, 3, 6, 7, 10, 11}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{4, 7, 12, 15}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{21, 22, 25, 26, 37, 38, 41, 42}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{0, 2, 8, 10, 32, 34, 40, 42}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{0, 3, 8, 11, 32, 35, 40, 43}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, scalar_constant_float32)
{
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), Shape{});
    auto r = op::Constant::create(element::Float32::element_type(), Shape{}, {4.8});
    auto f = make_shared<Function>(r, rt, op::Parameters{});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), Shape{});

    cf->call({}, {result});
    EXPECT_EQ(vector<float>{4.8}, result->get_vector<float>());
}

TEST(${BACKEND_NAME}, scalar_constant_int64)
{
    auto rt = make_shared<TensorViewType>(element::Int64::element_type(), Shape{});
    auto r = op::Constant::create(element::Int64::element_type(), Shape{}, {2112});
    auto f = make_shared<Function>(r, rt, op::Parameters{});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_primary_tensor_view(element::Int64::element_type(), Shape{});

    cf->call({}, {result});
    EXPECT_EQ(vector<int64_t>{{2112}}, result->get_vector<int64_t>());
}

TEST(${BACKEND_NAME}, tensor_constant_float32)
{
    auto shape = Shape{2, 2};
    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto r = op::Constant::create(element::Float32::element_type(), shape, {4.8, 4.7, -5.3, 0.0});
    auto f = make_shared<Function>(r, rt, op::Parameters{});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    cf->call({}, {result});
    EXPECT_EQ((vector<float>{4.8, 4.7, -5.3, 0}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, tensor_constant_int64)
{
    auto shape = Shape{2, 2};
    auto rt = make_shared<TensorViewType>(element::Int64::element_type(), shape);
    auto r = op::Constant::create(element::Int64::element_type(), shape, {2112, 1848, 1776, 1964});
    auto f = make_shared<Function>(r, rt, op::Parameters{});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_primary_tensor_view(element::Int64::element_type(), shape);

    cf->call({}, {result});
    EXPECT_EQ((vector<int64_t>{2112, 1848, 1776, 1964}), result->get_vector<int64_t>());
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
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
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
    EXPECT_EQ((vector<float>{10}), result->get_vector<float>());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), a->get_vector<float>());
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
    EXPECT_EQ((vector<float>{9, 12}), result->get_vector<float>());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), a->get_vector<float>());
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
    EXPECT_EQ((vector<float>{3, 7, 11}), result->get_vector<float>());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), a->get_vector<float>());
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
    copy_data(result, vector<float>({3, 3, 3}));

    cf->call({a}, {result});
    EXPECT_EQ((vector<float>{0, 0, 0}), result->get_vector<float>());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), a->get_vector<float>());
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
    copy_data(result, vector<float>({3, 3}));

    cf->call({a}, {result});
    EXPECT_EQ((vector<float>{0, 0}), result->get_vector<float>());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), a->get_vector<float>());
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
    copy_data(result, vector<float>({3}));

    cf->call({a}, {result});
    EXPECT_EQ((vector<float>{0}), result->get_vector<float>());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), a->get_vector<float>());
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
    copy_data(result, vector<float>({3}));

    cf->call({a}, {result});
    EXPECT_EQ((vector<float>{0}), result->get_vector<float>());

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), a->get_vector<float>());
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
    EXPECT_EQ((vector<float>{1 + 10 + 19,
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
    EXPECT_EQ((vector<float>{1 + 2 + 3,
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
    EXPECT_EQ((vector<float>{1 + 10 + 19 + 4 + 13 + 22 + 7 + 16 + 25,
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
    EXPECT_EQ((vector<float>{1 + 10 + 19 + 4 + 13 + 22 + 7 + 16 + 25 + 2 + 11 + 20 + 5 + 14 + 23 +
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
    EXPECT_EQ((vector<float>{0, 0, 0, 0, 0, 0}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{1, -1, 0, -1, 1, 0}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{1, 1, 729, 125}), result->get_vector<float>());
}

TEST(${BACKEND_NAME}, constant_equality_bool)
{
    auto shape = Shape{4};
    // auto A = make_shared<op::Parameter>(element::Bool::element_type(), shape);
    // auto B = make_shared<op::Parameter>(element::Bool::element_type(), shape);
    // auto result_type = make_shared<TensorViewType>(element::Bool::element_type(), shape);
    // auto f = make_shared<Function>(make_shared<op::Equal>(A, B), result_type, op::Parameters{A, B});

    auto A = op::Constant::create(element::boolean, shape, {true, false, true, false});
    auto B = op::Constant::create(element::boolean, shape, {true, true, true, true});
    auto rt = make_shared<TensorViewType>(element::Bool::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Equal>(A, B), rt, op::Parameters{});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto result = backend->make_primary_tensor_view(element::Bool::element_type(), shape);

    cf->call({}, {result});
    EXPECT_EQ((vector<char>{true, false, true, false}), result->get_vector<char>());
}

TEST(${BACKEND_NAME}, sqrt)
{
    auto shape = Shape{2, 3};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto result_type = make_shared<TensorViewType>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(make_shared<op::Sqrt>(A), result_type, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{16, 4, 81, 100, 10000, 0});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    cf->call({a}, {result});
    EXPECT_EQ((vector<float>{4, 2, 9, 10, 100, 0}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{808}), result->get_vector<float>());
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
    EXPECT_EQ((vector<float>{1, 102, 103, 4, 5, 106, 107, 8, 9, 110, 111, 12, 13, 14, 15, 16}),
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
    EXPECT_EQ(
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
    EXPECT_EQ((vector<int32_t>{0, 0, 1}), result->get_vector<int32_t>());
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
    EXPECT_EQ((vector<int32_t>{0, 1, 0}), result->get_vector<int32_t>());
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
    EXPECT_EQ((vector<int32_t>{1, 0, 0}), result->get_vector<int32_t>());
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
    EXPECT_EQ(
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
    EXPECT_EQ(
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

TEST(${BACKEND_NAME}, one_hot_matrix_0)
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
    EXPECT_EQ((vector<int32_t>{1, 0, 0, 0, 0, 1, 1, 0, 0,

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
    EXPECT_EQ(
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
    EXPECT_EQ((vector<float>{0,  1,  2,  3,  4,  5,   6,   7,  8,  9,   10,  11, 12, 13, 14, 15,

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
    EXPECT_EQ((vector<float>{900, 1,  902, 3,  4,  5,  6,  7,  908, 9,  910, 11, 12, 13, 14, 15,

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
    EXPECT_EQ((vector<float>{900, 1,  2,  903, 4,  5,  6,  7,  908, 9,  10, 911, 12, 13, 14, 15,

                             16,  17, 18, 19,  20, 21, 22, 23, 24,  25, 26, 27,  28, 29, 30, 31,

                             932, 33, 34, 935, 36, 37, 38, 39, 940, 41, 42, 943, 44, 45, 46, 47,

                             48,  49, 50, 51,  52, 53, 54, 55, 56,  57, 58, 59,  60, 61, 62, 63}),
              result->get_vector<float>());
}

//
// Numpy test:
//
// > from numpy import *
// > x = linspace(1,2*3*4,2*3*4)
// > y = linspace(1,3*4*5,3*4*5)
// > x.shape=(2,3,4)
// > y.shape=(3,4,5)
// > z = tensordot(x,y,([1,2],[0,1]))
// > z.shape = 2*5
// > z
// array([ 2938.,  3016.,  3094.,  3172.,  3250.,  7042.,  7264.,  7486.,
//         7708.,  7930.])
//
// Disabled because it doesn't work on CPU yet.
//
TEST(DISABLED_${BACKEND_NAME}, dot_3d_multi_axis)
{
    vector<float> a_data(2 * 3 * 4);
    for (int i = 0; i < 2 * 3 * 4; i++)
    {
        a_data[i] = float(i + 1);
    }

    vector<float> b_data(3 * 4 * 5);
    for (int i = 0; i < 3 * 4 * 5; i++)
    {
        b_data[i] = float(i + 1);
    }

    auto shape_a = Shape{2, 3, 4};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_b = Shape{3, 4, 5};
    auto B = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_b));
    auto shape_r = Shape{2, 5};

    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Dot>(A, B, 2);
    auto f = make_shared<Function>(r, rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, a_data);
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape_b);
    copy_data(b, b_data);

    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a, b}, {result});
    EXPECT_EQ((vector<float>{2938., 3016., 3094., 3172., 3250., 7042., 7264., 7486., 7708., 7930.}),
              result->get_vector<float>());
}

//
// Numpy test:
//
// > from numpy import *
// > x = array([6,61,2,3,5,21,75,23,23,0,23,2,35,67,1,2,9,16,2,3,6,1,8,0])
// > y = array([9,1,4,6,3,5,1,36,7,3,5,0,1,20,35,2,1,0,1,25,3,6,7,8])
// > x.shape=(2,4,3)
// > y.shape=(3,4,2)
// > z = tensordot(x,y,([2],[0]))
// > z.shape = 2*4*4*2
// > z
// array([ 483,  189,  331,   86,   85, 1262, 2155,  354,   83,   18,   58,
//         543,   77,  241,  325,  286,  859,  144,  438, 1025,  317,  973,
//        1041, 2930,  163,   69,  117,   50,   29,  472,  819,   62,  785,
//         236,  476,  235,  175, 1521, 2387, 1402,   97,   29,   69,  412,
//          63,  286,  429,  218,   45,   11,   29,  162,   27,  106,  149,
//         126,   65,   25,   44,    6,   11,  165,  281,   52])
//
// Disabled because it doesn't work on CPU yet.
//
TEST(DISABLED_${BACKEND_NAME}, dot_3d_one_axis_arbitrary)
{
    vector<float> a_data{6,  61, 2, 3, 5, 21, 75, 23, 23, 0, 23, 2,
                         35, 67, 1, 2, 9, 16, 2,  3,  6,  1, 8,  0};
    vector<float> b_data{9, 1,  4,  6, 3, 5, 1, 36, 7, 3, 5, 0,
                         1, 20, 35, 2, 1, 0, 1, 25, 3, 6, 7, 8};

    auto shape_a = Shape{2, 4, 3};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_b = Shape{3, 4, 2};
    auto B = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_b));
    auto shape_r = Shape{2, 4, 4, 2};

    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Dot>(A, B);
    auto f = make_shared<Function>(r, rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, a_data);
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape_b);
    copy_data(b, b_data);

    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a, b}, {result});
    EXPECT_EQ((vector<float>{483,  189, 331, 86,  85,  1262, 2155, 354, 83,  18,   58,   543,  77,
                             241,  325, 286, 859, 144, 438,  1025, 317, 973, 1041, 2930, 163,  69,
                             117,  50,  29,  472, 819, 62,   785,  236, 476, 235,  175,  1521, 2387,
                             1402, 97,  29,  69,  412, 63,   286,  429, 218, 45,   11,   29,   162,
                             27,   106, 149, 126, 65,  25,   44,   6,   11,  165,  281,  52}),
              result->get_vector<float>());
}

//
// Numpy test:
//
// from numpy import *
// x = linspace(1,2*3*3*4,2*3*3*4)
// y = linspace(1,3*4*2*3*2,3*4*2*2*3)
// x.shape=(2,3,3,4)
// y.shape=(3,4,2,2,3)
// z = tensordot(x,y,([2,3],[0,1]))
// z.shape = 2*3*2*2*3
// z
//
// array([  6942.,   7020.,   7098.,   7176.,   7254.,   7332.,   7410.,
//          7488.,   7566.,   7644.,   7722.,   7800.,  16590.,  16812.,
//         17034.,  17256.,  17478.,  17700.,  17922.,  18144.,  18366.,
//         18588.,  18810.,  19032.,  26238.,  26604.,  26970.,  27336.,
//         27702.,  28068.,  28434.,  28800.,  29166.,  29532.,  29898.,
//         30264.,  35886.,  36396.,  36906.,  37416.,  37926.,  38436.,
//         38946.,  39456.,  39966.,  40476.,  40986.,  41496.,  45534.,
//         46188.,  46842.,  47496.,  48150.,  48804.,  49458.,  50112.,
//         50766.,  51420.,  52074.,  52728.,  55182.,  55980.,  56778.,
//         57576.,  58374.,  59172.,  59970.,  60768.,  61566.,  62364.,
//         63162.,  63960.])
//
// Disabled because it doesn't work on CPU yet.
//
TEST(DISABLED_${BACKEND_NAME}, dot_4d_5d_multi_axis)
{
    vector<float> a_data(2 * 3 * 3 * 4);
    for (int i = 0; i < 2 * 3 * 3 * 4; i++)
    {
        a_data[i] = float(i + 1);
    }

    vector<float> b_data(3 * 4 * 2 * 2 * 3);
    for (int i = 0; i < 3 * 4 * 2 * 2 * 3; i++)
    {
        b_data[i] = float(i + 1);
    }

    auto shape_a = Shape{2, 3, 3, 4};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_b = Shape{3, 4, 2, 3, 2};
    auto B = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_b));
    auto shape_r = Shape{2, 3, 2, 3, 2};

    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Dot>(A, B, 2);
    auto f = make_shared<Function>(r, rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, a_data);
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape_b);
    copy_data(b, b_data);

    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a, b}, {result});
    EXPECT_EQ(
        (vector<float>{6942.,  7020.,  7098.,  7176.,  7254.,  7332.,  7410.,  7488.,  7566.,
                       7644.,  7722.,  7800.,  16590., 16812., 17034., 17256., 17478., 17700.,
                       17922., 18144., 18366., 18588., 18810., 19032., 26238., 26604., 26970.,
                       27336., 27702., 28068., 28434., 28800., 29166., 29532., 29898., 30264.,
                       35886., 36396., 36906., 37416., 37926., 38436., 38946., 39456., 39966.,
                       40476., 40986., 41496., 45534., 46188., 46842., 47496., 48150., 48804.,
                       49458., 50112., 50766., 51420., 52074., 52728., 55182., 55980., 56778.,
                       57576., 58374., 59172., 59970., 60768., 61566., 62364., 63162., 63960.}),
        result->get_vector<float>());
}

//
// Numpy test:
//
// from numpy import *
// x = linspace(1,2*3*3*4,2*3*3*4)
// y = linspace(1,2*3*3*4*2,2*3*3*4*2)
// x.shape=(2,3,3,4)
// y.shape=(2,3,3,4,2)
// z = tensordot(x,y,([0,1,2,3],[0,1,2,3]))
// z
//
// array([ 251412.,  254040.])
//
// Disabled because it doesn't work on CPU yet.
//
TEST(DISABLED_${BACKEND_NAME}, dot_4d_5d_multi_axis_more)
{
    vector<float> a_data(2 * 3 * 3 * 4);
    for (int i = 0; i < 2 * 3 * 3 * 4; i++)
    {
        a_data[i] = float(i + 1);
    }

    vector<float> b_data(2 * 3 * 3 * 4 * 2);
    for (int i = 0; i < 2 * 3 * 3 * 4 * 2; i++)
    {
        b_data[i] = float(i + 1);
    }

    auto shape_a = Shape{2, 3, 3, 4};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_a));
    auto shape_b = Shape{2, 3, 3, 4, 2};
    auto B = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float32::element_type(), shape_b));
    auto shape_r = Shape{2};

    auto rt = make_shared<TensorViewType>(element::Float32::element_type(), shape_r);
    auto r = make_shared<op::Dot>(A, B, 4);
    auto f = make_shared<Function>(r, rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape_a);
    copy_data(a, a_data);
    auto b = backend->make_primary_tensor_view(element::Float32::element_type(), shape_b);
    copy_data(b, b_data);

    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape_r);

    cf->call({a, b}, {result});
    EXPECT_EQ((vector<float>{251412., 254040.}), result->get_vector<float>());
}

//
// Numpy test:
//
// from numpy import *
// x = linspace(1,20*30*30*40,20*30*30*40)
// y = linspace(1,20*30*30*40*20,20*30*30*40*20)
// x.shape=(20,30,30,40)
// y.shape=(20,30,30,40,20)
// z = tensordot(x,y,([0,1,2,3],[0,1,2,3]))
// set_printoptions(precision=20)
// z
//
// array([  2.48832025919525478400e+18,   2.48832051839533977600e+18,
//          2.48832077759658444800e+18,   2.48832103679413504000e+18,
//          2.48832129599669350400e+18,   2.48832155519793971200e+18,
//          2.48832181439802265600e+18,   2.48832207359808000000e+18,
//          2.48832233279813580800e+18,   2.48832259199822028800e+18,
//          2.48832285119946496000e+18,   2.48832311040043008000e+18,
//          2.48832336959957401600e+18,   2.48832362880081817600e+18,
//          2.48832388800090368000e+18,   2.48832414720096000000e+18,
//          2.48832440640101478400e+18,   2.48832466560109772800e+18,
//          2.48832492480234188800e+18,   2.48832518400031897600e+18])
//
// Disabled because this test is very slow.
//
TEST(DISABLED_${BACKEND_NAME}, dot_4d_5d_multi_axis_big_fp64_VERY_SLOW)
{
    vector<double> a_data(20 * 30 * 30 * 40);
    for (int i = 0; i < 20 * 30 * 30 * 40; i++)
    {
        a_data[i] = double(i + 1);
    }

    vector<double> b_data(20 * 30 * 30 * 40 * 20);
    for (int i = 0; i < 20 * 30 * 30 * 40 * 20; i++)
    {
        b_data[i] = double(i + 1);
    }

    auto shape_a = Shape{20, 30, 30, 40};
    auto A = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float64::element_type(), shape_a));
    auto shape_b = Shape{20, 30, 30, 40, 20};
    auto B = make_shared<op::Parameter>(
        make_shared<TensorViewType>(element::Float64::element_type(), shape_b));
    auto shape_r = Shape{20};

    auto rt = make_shared<TensorViewType>(element::Float64::element_type(), shape_r);
    auto r = make_shared<op::Dot>(A, B, 4);
    auto f = make_shared<Function>(r, rt, op::Parameters{A, B});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float64::element_type(), shape_a);
    copy_data(a, a_data);
    auto b = backend->make_primary_tensor_view(element::Float64::element_type(), shape_b);
    copy_data(b, b_data);

    auto result = backend->make_primary_tensor_view(element::Float64::element_type(), shape_r);

    cf->call({a, b}, {result});
    EXPECT_TRUE(test::all_close(
        vector<double>{
            2.48832025919525478400e+18, 2.48832051839533977600e+18, 2.48832077759658444800e+18,
            2.48832103679413504000e+18, 2.48832129599669350400e+18, 2.48832155519793971200e+18,
            2.48832181439802265600e+18, 2.48832207359808000000e+18, 2.48832233279813580800e+18,
            2.48832259199822028800e+18, 2.48832285119946496000e+18, 2.48832311040043008000e+18,
            2.48832336959957401600e+18, 2.48832362880081817600e+18, 2.48832388800090368000e+18,
            2.48832414720096000000e+18, 2.48832440640101478400e+18, 2.48832466560109772800e+18,
            2.48832492480234188800e+18, 2.48832518400031897600e+18},
        result->get_vector<double>()));
}

TEST(${BACKEND_NAME}, DISABLED_parameter_to_output)
{
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto f = make_shared<Function>(A, op::Parameters{A});

    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
    auto cf = backend->make_call_frame(external);

    // Create some tensors for input/output
    auto a = backend->make_primary_tensor_view(element::Float32::element_type(), shape);
    copy_data(a, vector<float>{1, -2, 0, -4.8f});
    auto result = backend->make_primary_tensor_view(element::Float32::element_type(), shape);

    cf->call({a}, {result});
    EXPECT_EQ((vector<float>{1, -2, 0, -4.8f}), result->get_vector<float>());
}
