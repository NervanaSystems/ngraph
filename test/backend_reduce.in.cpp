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

// Trivial case with no reduction axes.
NGRAPH_TEST(${BACKEND_NAME}, reduce_trivial)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), op::ParameterVector{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape{2, 2};
    auto g_A = make_shared<op::Parameter>(element::f32, shape);
    auto g_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto g = make_shared<Function>(make_shared<op::Reduce>(g_A, g_B, f, AxisSet{}),
                                   op::ParameterVector{g_A, g_B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto b = backend->create_tensor(element::f32, {});
    copy_data(b, vector<float>{0});
    auto result = backend->create_tensor(element::f32, shape);

    backend->call_with_validate(g, {result}, {a, b});
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_to_scalar)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), op::ParameterVector{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape{2, 2};
    auto g_A = make_shared<op::Parameter>(element::f32, shape);
    auto g_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto g = make_shared<Function>(make_shared<op::Reduce>(g_A, g_B, f, AxisSet{0, 1}),
                                   op::ParameterVector{g_A, g_B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape);
    copy_data(a, vector<float>{1, 2, 3, 4});
    auto b = backend->create_tensor(element::f32, Shape{});
    copy_data(b, vector<float>{0});
    auto result = backend->create_tensor(element::f32, Shape{});

    backend->call_with_validate(g, {result}, {a, b});
    EXPECT_EQ((vector<float>{10}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4}), read_vector<float>(a));
    EXPECT_EQ((vector<float>{0}), read_vector<float>(b));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_matrix_columns)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});

    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), op::ParameterVector{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{3, 2};
    auto g_A = make_shared<op::Parameter>(element::f32, shape_a);
    auto g_B = make_shared<op::Parameter>(element::f32, Shape{});
    Shape shape_rt{2};

    auto g = make_shared<Function>(make_shared<op::Reduce>(g_A, g_B, f, AxisSet{0}),
                                   op::ParameterVector{g_A, g_B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto b = backend->create_tensor(element::f32, Shape{});
    copy_data(b, vector<float>{0});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call_with_validate(g, {result}, {a, b});
    EXPECT_EQ((vector<float>{9, 12}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
    EXPECT_EQ((vector<float>{0}), read_vector<float>(b));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_matrix_rows)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});

    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), op::ParameterVector{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{3, 2};
    auto g_A = make_shared<op::Parameter>(element::f32, shape_a);
    auto g_B = make_shared<op::Parameter>(element::f32, Shape{});
    Shape shape_rt{3};
    auto g = make_shared<Function>(make_shared<op::Reduce>(g_A, g_B, f, AxisSet{1}),
                                   op::ParameterVector{g_A, g_B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1, 2, 3, 4, 5, 6});
    auto b = backend->create_tensor(element::f32, Shape{});
    copy_data(b, vector<float>{0});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call_with_validate(g, {result}, {a, b});
    EXPECT_EQ((vector<float>{3, 7, 11}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{1, 2, 3, 4, 5, 6}), read_vector<float>(a));
    EXPECT_EQ((vector<float>{0}), read_vector<float>(b));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_matrix_rows_zero)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), op::ParameterVector{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{3, 0};
    auto g_A = make_shared<op::Parameter>(element::f32, shape_a);
    auto g_B = make_shared<op::Parameter>(element::f32, Shape{});
    Shape shape_rt{3};
    auto g = make_shared<Function>(make_shared<op::Reduce>(g_A, g_B, f, AxisSet{1}),
                                   op::ParameterVector{g_A, g_B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->create_tensor(element::f32, Shape{});
    copy_data(b, vector<float>{66});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call_with_validate(g, {result}, {a, b});
    EXPECT_EQ((vector<float>{66, 66, 66}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
    EXPECT_EQ((vector<float>{66}), read_vector<float>(b));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_matrix_cols_zero)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), op::ParameterVector{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{0, 2};
    auto g_A = make_shared<op::Parameter>(element::f32, shape_a);
    auto g_B = make_shared<op::Parameter>(element::f32, Shape{});
    Shape shape_rt{2};
    auto g = make_shared<Function>(make_shared<op::Reduce>(g_A, g_B, f, AxisSet{0}),
                                   op::ParameterVector{g_A, g_B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->create_tensor(element::f32, Shape{});
    copy_data(b, vector<float>{77});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call_with_validate(g, {result}, {a, b});
    EXPECT_EQ((vector<float>{77, 77}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
    EXPECT_EQ((vector<float>{77}), read_vector<float>(b));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_vector_zero)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), op::ParameterVector{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{0};
    auto g_A = make_shared<op::Parameter>(element::f32, shape_a);
    auto g_B = make_shared<op::Parameter>(element::f32, Shape{});
    Shape shape_rt{};
    auto g = make_shared<Function>(make_shared<op::Reduce>(g_A, g_B, f, AxisSet{0}),
                                   op::ParameterVector{g_A, g_B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->create_tensor(element::f32, Shape{});
    copy_data(b, vector<float>{88});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call_with_validate(g, {result}, {a, b});
    EXPECT_EQ((vector<float>{88}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
    EXPECT_EQ((vector<float>{88}), read_vector<float>(b));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_matrix_to_scalar_zero_by_zero)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x+y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto f = make_shared<Function>(make_shared<op::Add>(f_A, f_B), op::ParameterVector{f_A, f_B});

    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{0, 0};
    auto g_A = make_shared<op::Parameter>(element::f32, shape_a);
    auto g_B = make_shared<op::Parameter>(element::f32, Shape{});
    Shape shape_rt{};
    auto g = make_shared<Function>(make_shared<op::Reduce>(g_A, g_B, f, AxisSet{0, 1}),
                                   op::ParameterVector{g_A, g_B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{});
    auto b = backend->create_tensor(element::f32, Shape{});
    copy_data(b, vector<float>{99});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call_with_validate(g, {result}, {a, b});
    EXPECT_EQ((vector<float>{99}), read_vector<float>(result));

    // For some reason I'm feeling extra paranoid about making sure reduction doesn't clobber the
    // input tensors, so let's do this too.
    EXPECT_EQ((vector<float>{}), read_vector<float>(a));
    EXPECT_EQ((vector<float>{99}), read_vector<float>(b));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_3d_to_vector)
{
    // First, the reduction function (f(x:float32[],y:float32[]) = x*y).
    auto f_A = make_shared<op::Parameter>(element::f32, Shape{});
    auto f_B = make_shared<op::Parameter>(element::f32, Shape{});
    auto f =
        make_shared<Function>(make_shared<op::Multiply>(f_A, f_B), op::ParameterVector{f_A, f_B});

    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_rt{3};
    auto g = make_shared<Function>(make_shared<op::Reduce>(A, B, f, AxisSet{0, 1}),
                                   op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a, vector<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                               15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27});
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(b, vector<float>{1});
    auto result = backend->create_tensor(element::f32, shape_rt);

    backend->call_with_validate(g, {result}, {a, b});
    EXPECT_EQ((vector<float>{1.0f * 10.0f * 19.0f * 4.0f * 13.0f * 22.0f * 7.0f * 16.0f * 25.0f,
                             2.0f * 11.0f * 20.0f * 5.0f * 14.0f * 23.0f * 8.0f * 17.0f * 26.0f,
                             3.0f * 12.0f * 21.0f * 6.0f * 15.0f * 24.0f * 9.0f * 18.0f * 27.0f}),
              read_vector<float>(result));
}

//
// The unit tests for ReduceWindow follow exactly what we test for MaxPool---but they use ReduceWindow to do it.
//
NGRAPH_TEST(${BACKEND_NAME}, reduce_window_emulating_max_pool_1d_1channel_1image)
{
    Shape shape_ra{};
    auto RA = make_shared<op::Parameter>(element::f32, shape_ra);
    Shape shape_rb{};
    auto RB = make_shared<op::Parameter>(element::f32, shape_rb);
    auto rf = make_shared<Function>(make_shared<op::Maximum>(RA, RB), op::ParameterVector{RA, RB});

    Shape shape_a{1, 1, 14};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{1, 1, 12};
    Shape window_shape{1, 1, 3};
    auto window_movement_strides = Strides{1, 1, 1};
    auto f = make_shared<Function>(
        make_shared<op::ReduceWindow>(A, B, rf, window_shape, window_movement_strides),
        op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 3>{{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}}}.get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(
        b,
        vector<float>{
            -1}); // Really should use -inf but since we know the values in the test vector this should work
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 3>({{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}}}).get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_window_emulating_max_pool_1d_1channel_2image)
{
    Shape shape_ra{};
    auto RA = make_shared<op::Parameter>(element::f32, shape_ra);
    Shape shape_rb{};
    auto RB = make_shared<op::Parameter>(element::f32, shape_rb);
    auto rf = make_shared<Function>(make_shared<op::Maximum>(RA, RB), op::ParameterVector{RA, RB});

    Shape shape_a{2, 1, 14};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{2, 1, 12};
    Shape window_shape{1, 1, 3};
    auto window_movement_strides = Strides{1, 1, 1};
    auto f = make_shared<Function>(
        make_shared<op::ReduceWindow>(A, B, rf, window_shape, window_movement_strides),
        op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0}},
                                       {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2}}})
                  .get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(
        b,
        vector<float>{
            -1}); // Really should use -inf but since we know the values in the test vector this should work
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 3>(
                   {{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}}, {{2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 1, 2}}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_window_emulating_max_pool_1d_2channel_2image)
{
    Shape shape_ra{};
    auto RA = make_shared<op::Parameter>(element::f32, shape_ra);
    Shape shape_rb{};
    auto RB = make_shared<op::Parameter>(element::f32, shape_rb);
    auto rf = make_shared<Function>(make_shared<op::Maximum>(RA, RB), op::ParameterVector{RA, RB});

    Shape shape_a{2, 2, 14};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{2, 2, 12};
    Shape window_shape{1, 1, 3};
    auto window_movement_strides = Strides{1, 1, 1};
    auto f = make_shared<Function>(
        make_shared<op::ReduceWindow>(A, B, rf, window_shape, window_movement_strides),
        op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 3>({{{0, 1, 0, 2, 1, 0, 3, 2, 0, 0, 2, 0, 0, 0},
                                        {0, 0, 0, 2, 0, 0, 2, 3, 0, 1, 2, 0, 1, 0}},

                                       {{0, 2, 1, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 2},
                                        {2, 1, 0, 0, 1, 0, 2, 0, 0, 0, 1, 1, 2, 0}}})
                  .get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(
        b,
        vector<float>{
            -1}); // Really should use -inf but since we know the values in the test vector this should work
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 3>(
                   {{{1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 2, 0}, {0, 2, 2, 2, 2, 3, 3, 3, 2, 2, 2, 1}},

                    {{2, 2, 1, 1, 0, 2, 2, 2, 1, 1, 1, 2}, {2, 1, 1, 1, 2, 2, 2, 0, 1, 1, 2, 2}}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_window_emulating_max_pool_2d_2channel_2image)
{
    Shape shape_ra{};
    auto RA = make_shared<op::Parameter>(element::f32, shape_ra);
    Shape shape_rb{};
    auto RB = make_shared<op::Parameter>(element::f32, shape_rb);
    auto rf = make_shared<Function>(make_shared<op::Maximum>(RA, RB), op::ParameterVector{RA, RB});

    Shape shape_a{2, 2, 5, 5};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{2, 2, 4, 3};
    Shape window_shape{1, 1, 2, 3};
    auto window_movement_strides = Strides{1, 1, 1, 1};
    auto f = make_shared<Function>(
        make_shared<op::ReduceWindow>(A, B, rf, window_shape, window_movement_strides),
        op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2, 1}, // img 0 chan 0
                                         {0, 3, 2, 0, 0},
                                         {2, 0, 0, 0, 1},
                                         {2, 0, 1, 1, 2},
                                         {0, 2, 1, 0, 0}},

                                        {{0, 0, 0, 2, 0}, // img 0 chan 1
                                         {0, 2, 3, 0, 1},
                                         {2, 0, 1, 0, 2},
                                         {3, 1, 0, 0, 0},
                                         {2, 0, 0, 0, 0}}},

                                       {{{0, 2, 1, 1, 0}, // img 1 chan 0
                                         {0, 0, 2, 0, 1},
                                         {0, 0, 1, 2, 3},
                                         {2, 0, 0, 3, 0},
                                         {0, 0, 0, 0, 0}},

                                        {{2, 1, 0, 0, 1}, // img 1 chan 1
                                         {0, 2, 0, 0, 0},
                                         {1, 1, 2, 0, 2},
                                         {1, 1, 1, 0, 1},
                                         {1, 0, 0, 0, 2}}}})
                  .get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(
        b,
        vector<float>{
            -1}); // Really should use -inf but since we know the values in the test vector this should work
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 4>({{{{3, 3, 2}, // img 0 chan 0
                                          {3, 3, 2},
                                          {2, 1, 2},
                                          {2, 2, 2}},

                                         {{3, 3, 3}, // img 0 chan 1
                                          {3, 3, 3},
                                          {3, 1, 2},
                                          {3, 1, 0}}},

                                        {{{2, 2, 2}, // img 1 chan 0
                                          {2, 2, 3},
                                          {2, 3, 3},
                                          {2, 3, 3}},

                                         {{2, 2, 1}, // img 1 chan 1
                                          {2, 2, 2},
                                          {2, 2, 2},
                                          {1, 1, 2}}}})
                   .get_vector()),
              read_vector<float>(result));
}

NGRAPH_TEST(${BACKEND_NAME}, reduce_window_emulating_max_pool_2d_1channel_1image_strided)
{
    Shape shape_ra{};
    auto RA = make_shared<op::Parameter>(element::f32, shape_ra);
    Shape shape_rb{};
    auto RB = make_shared<op::Parameter>(element::f32, shape_rb);
    auto rf = make_shared<Function>(make_shared<op::Maximum>(RA, RB), op::ParameterVector{RA, RB});

    Shape shape_a{1, 1, 8, 8};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_b{};
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    Shape shape_r{1, 1, 3, 3};
    Shape window_shape{1, 1, 2, 3};
    auto window_movement_strides = Strides{1, 1, 3, 2};
    auto f = make_shared<Function>(
        make_shared<op::ReduceWindow>(A, B, rf, window_shape, window_movement_strides),
        op::ParameterVector{A, B});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, shape_a);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2, 1, 2, 0, 0},
                                         {0, 3, 2, 0, 0, 0, 1, 0},
                                         {2, 0, 0, 0, 1, 0, 0, 0},
                                         {2, 0, 1, 1, 2, 2, 3, 0},
                                         {0, 2, 1, 0, 0, 0, 1, 0},
                                         {2, 0, 3, 1, 0, 0, 0, 0},
                                         {1, 2, 0, 0, 0, 1, 2, 0},
                                         {1, 0, 2, 0, 0, 0, 1, 0}}}})
                  .get_vector());
    auto b = backend->create_tensor(element::f32, shape_b);
    copy_data(
        b,
        vector<float>{
            -1}); // Really should use -inf but since we know the values in the test vector this should work
    auto result = backend->create_tensor(element::f32, shape_r);

    backend->call_with_validate(f, {result}, {a, b});
    EXPECT_EQ((test::NDArray<float, 4>({{{{3, 2, 2}, {2, 2, 3}, {2, 2, 2}}}}).get_vector()),
              read_vector<float>(result));
}
