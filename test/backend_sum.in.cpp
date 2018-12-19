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
#include "util/random.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

static std::mt19937_64 random_generator;

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

// Trivial case with no summed axes.
NGRAPH_TEST(${BACKEND_NAME}, sum_trivial)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(result));
}

// Failure has been reported at 5D for some reason
NGRAPH_TEST(${BACKEND_NAME}, sum_trivial_5d)
{
    Shape shape{2, 2, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto result = backend->create_tensor(element::f32, shape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_to_scalar)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0, 1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto result = backend->create_tensor(element::f32, Shape{});

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{10}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_large_1d_to_scalar)
{
    Shape shape{1000000};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    random_generator.seed(2);
    vector<float> v_a(1000000, 0);
    double r = 0;
    for (int i = 0; i < 1000000; i++)
    {
        v_a[i] = static_cast<float>(random_generator() % 255);
        r += static_cast<double>(v_a[i]);
    }
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, v_a);
    auto result = backend->create_tensor(element::f32, Shape{});

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});

    EXPECT_TRUE(
        test::all_close_f(vector<float>{static_cast<float>(r)}, read_vector<float>(result)));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_columns)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{9, 12}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_6d)
{
    Shape shape_a{2, 6, 4, 5, 7, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2, 4, 5, 3};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{1, 4}), ParameterVector{A});

    auto backend_wrk = runtime::Backend::create("${BACKEND_NAME}");
    auto backend_ref = runtime::Backend::create("INTERPRETER");

    // Create some tensors for input/output
    auto a_wrk = backend_wrk->create_tensor(element::f32, shape_a);
    auto a_ref = backend_ref->create_tensor(element::f32, shape_a);
    auto result_wrk = backend_wrk->create_tensor(element::f32, shape_rt);
    auto result_ref = backend_ref->create_tensor(element::f32, shape_rt);

    vector<float> inp_data(shape_size<const Shape>(shape_a));
    iota(inp_data.begin(), inp_data.end(), 1);
    copy_data(a_wrk, inp_data);
    copy_data(a_ref, inp_data);

    backend_wrk->call_with_validate(backend_wrk->compile(f), {result_wrk}, {a_wrk});
    backend_ref->call_with_validate(backend_ref->compile(f), {result_ref}, {a_ref});

    EXPECT_EQ(read_vector<float>(result_ref), read_vector<float>(result_wrk));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_rows)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{3, 7, 11}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_rows_zero)
{
    Shape shape_a{3, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3, 3}));

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{0, 0, 0}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_cols_zero)
{
    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3, 3}));

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{0, 0}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_vector_zero)
{
    Shape shape_a{0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{0}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_matrix_to_scalar_zero_by_zero)
{
    Shape shape_a{0, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0, 1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);
    copy_data(result, vector<float>({3}));

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{0}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_3d_to_matrix_most_sig)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{1 + 10 + 19,
                             2 + 11 + 20,
                             3 + 12 + 21,
                             4 + 13 + 22,
                             5 + 14 + 23,
                             6 + 15 + 24,
                             7 + 16 + 25,
                             8 + 17 + 26,
                             9 + 18 + 27}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_3d_to_matrix_least_sig)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{2}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{1 + 2 + 3,
                             4 + 5 + 6,
                             7 + 8 + 9,
                             10 + 11 + 12,
                             13 + 14 + 15,
                             16 + 17 + 18,
                             19 + 20 + 21,
                             22 + 23 + 24,
                             25 + 26 + 27}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_3d_to_vector)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0, 1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{1 + 10 + 19 + 4 + 13 + 22 + 7 + 16 + 25,
                             2 + 11 + 20 + 5 + 14 + 23 + 8 + 17 + 26,
                             3 + 12 + 21 + 6 + 15 + 24 + 9 + 18 + 27}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_3d_to_scalar)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0, 1, 2}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{1 + 10 + 19 + 4 + 13 + 22 + 7 + 16 + 25 + 2 + 11 + 20 + 5 + 14 + 23 +
                             8 + 17 + 26 + 3 + 12 + 21 + 6 + 15 + 24 + 9 + 18 + 27}),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_3d_eliminate_zero_dim)
{
    Shape shape_a{3, 0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 2};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto result = backend->create_tensor(element::f32, shape_rt);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the right value.
    copy_data(result, vector<float>{2112, 2112, 2112, 2112, 2112, 2112});

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<float>{0, 0, 0, 0, 0, 0}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_3d_eliminate_zero_dim_int32)
{
    Shape shape_a{3, 0, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_rt{3, 2};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, vector<int32_t>{});
    auto result = backend->create_tensor(element::i32, shape_rt);

    // Overwrite the initial result vector to make sure we're not just coincidentally getting the right value.
    copy_data(result, vector<int32_t>{2112, 2112, 2112, 2112, 2112, 2112});

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<int32_t>{0, 0, 0, 0, 0, 0}), read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_5d_to_scalar)
{
    Shape shape_a{3, 3, 3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f =
        make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0, 1, 2, 3, 4}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, std::vector<float>(std::pow(3, 5), 1));
    auto result = backend->create_tensor(element::f32, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ(std::vector<float>{243.}, read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_5d_to_scalar_int32)
{
    Shape shape_a{3, 3, 3, 3, 3};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_rt{};
    auto f =
        make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0, 1, 2, 3, 4}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i32, shape_a);
    copy_data(a, std::vector<int32_t>(std::pow(3, 5), 1));
    auto result = backend->create_tensor(element::i32, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ(std::vector<int32_t>{243}, read_vector<int32_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_2d_to_scalar_int8)
{
    Shape shape_a{3, 3};
    auto A = make_shared<op::Parameter>(element::i8, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0, 1}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::i8, shape_a);
    copy_data(a, std::vector<int8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    auto result = backend->create_tensor(element::i8, shape_rt);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ(std::vector<int8_t>{45}, read_vector<int8_t>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_trivial_in_double)
{
    Shape shape{4, 3};
    Shape rshape{3};
    auto A = make_shared<op::Parameter>(element::f64, shape);
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0}), ParameterVector{A});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f64, shape);
    copy_data(a, vector<double>{12, 2, 10, 9, 8, 4, 6, 1, 5, 3, 11, 7});
    auto result = backend->create_tensor(element::f64, rshape);

    auto handle = backend->compile(f);
    backend->call_with_validate(handle, {result}, {a});
    EXPECT_EQ((vector<double>{30, 22, 26}), read_vector<double>(result));
}

#if NGRAPH_INTERPRETER_ENABLE

NGRAPH_TEST(${BACKEND_NAME}, sum_stable_acc)
{
    std::string backend_name = "${BACKEND_NAME}";
    if (backend_name == "INTERPRETER")
    {
        return;
    }
    Shape shape_a{10, 10, 10, 30};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);

    Shape shape_rt{10};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{1, 2, 3}), ParameterVector{A});

    test::Uniform<float> rng(1000.0f, 1000.1f, 2112);
    vector<vector<float>> args;
    for (shared_ptr<op::Parameter> param : f->get_parameters())
    {
        vector<float> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }

    auto ref_func = clone_function(*f);
    auto bk_func = clone_function(*f);

    auto ref_results = execute(ref_func, args, "INTERPRETER");
    auto bk_results = execute(bk_func, args, "${BACKEND_NAME}");

    EXPECT_TRUE(test::all_close_f(ref_results.at(0), bk_results.at(0), 24, 3));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_stable_acc_double)
{
    std::string backend_name = "${BACKEND_NAME}";
    if (backend_name == "INTERPRETER")
    {
        return;
    }
    Shape shape_a{10, 10, 20, 300};
    auto A = make_shared<op::Parameter>(element::f64, shape_a);

    Shape shape_rt{10};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{1, 2, 3}), ParameterVector{A});

    test::Uniform<double> rng(1000000000.0L, 1000000000.001L, 2112);
    vector<vector<double>> args;
    for (shared_ptr<op::Parameter> param : f->get_parameters())
    {
        vector<double> tensor_val(shape_size(param->get_shape()));
        rng.initialize(tensor_val);
        args.push_back(tensor_val);
    }

    auto ref_func = clone_function(*f);
    auto bk_func = clone_function(*f);

    auto ref_results = execute(ref_func, args, "INTERPRETER");
    auto bk_results = execute(bk_func, args, "${BACKEND_NAME}");

    EXPECT_TRUE(test::all_close(ref_results.at(0), bk_results.at(0), 0.0, 1e-5));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_stable_simple_float)
{
    std::string backend_name = "${BACKEND_NAME}";
    if (backend_name == "INTERPRETER")
    {
        return;
    }
    Shape shape_a{20};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);

    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0}), ParameterVector{A});

    vector<vector<float>> args;
    args.push_back(vector<float>{10000000.0f, 0.9f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f,
                                 0.8f,        0.1f, 0.9f, 0.5f, 0.2f, 0.3f, 0.4f,
                                 0.5f,        0.6f, 0.7f, 0.8f, 0.9f, 0.1f});

    auto ref_func = clone_function(*f);
    auto bk_func = clone_function(*f);

    auto ref_results = execute(ref_func, args, "INTERPRETER");
    auto bk_results = execute(bk_func, args, "${BACKEND_NAME}");

    EXPECT_TRUE(test::all_close_f(ref_results.at(0), bk_results.at(0), 24, 1));
}

NGRAPH_TEST(${BACKEND_NAME}, sum_stable_simple_double)
{
    std::string backend_name = "${BACKEND_NAME}";
    if (backend_name == "INTERPRETER")
    {
        return;
    }
    Shape shape_a{20};
    auto A = make_shared<op::Parameter>(element::f64, shape_a);

    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Sum>(A, AxisSet{0}), ParameterVector{A});

    vector<vector<double>> args;
    args.push_back(vector<double>{10000000000000000.0L,
                                  0.2L,
                                  0.3L,
                                  0.4L,
                                  0.5L,
                                  0.6L,
                                  0.7L,
                                  0.8L,
                                  0.9L,
                                  0.7L,
                                  0.9L,
                                  0.7L,
                                  0.3L,
                                  0.6L,
                                  0.8L,
                                  0.4L,
                                  0.6L,
                                  0.5L,
                                  0.8L,
                                  0.7L});

    auto ref_func = clone_function(*f);
    auto bk_func = clone_function(*f);

    auto ref_results = execute(ref_func, args, "INTERPRETER");
    auto bk_results = execute(bk_func, args, "${BACKEND_NAME}");

    EXPECT_TRUE(test::all_close(ref_results.at(0), bk_results.at(0), 0.0, 2.0));
}
#endif
