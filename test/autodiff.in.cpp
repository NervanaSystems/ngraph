/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <algorithm>
#include <functional>
#include <memory>
#include <tuple>

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/reference/avg_pool.hpp"
#include "util/autodiff/backprop_function.hpp"
#include "util/autodiff/numeric_compare.hpp"
#include "util/random.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

NGRAPH_TEST(${BACKEND_NAME}, backwards_maxpool_n4_c1_hw4_2x2_max)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape_a{1, 4, 4, 4}; //in CHWN
    Shape maxpool_shape{1, 4, 3, 3};

    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    auto reshape = make_shared<op::Reshape>(
        A, AxisVector{0, 3, 1, 2}, Shape{1, 4, 4, 4}); //convert CHWN to CNHW
    Shape window_shape{2, 2};
    auto window_movement_strides = Strides{1, 1};
    auto maxpool = make_shared<op::MaxPool>(reshape, window_shape, window_movement_strides);
    auto f = make_shared<Function>(maxpool, op::ParameterVector{A});

    shared_ptr<runtime::TensorView> ep = backend->create_tensor(element::i32, maxpool_shape);
    vector<int> dataEp(shape_size(maxpool_shape), 4);

    shared_ptr<runtime::TensorView> input = backend->create_tensor(element::i32, shape_a);
    shared_ptr<runtime::TensorView> output = backend->create_tensor(element::i32, shape_a);

    vector<int> dataInput{11, 65, 44, 28, 31, 33, 21, 66, 40, 49, 69, 57, 47, 30, 24, 27,
                          13, 56, 46, 60, 61, 41, 25, 42, 48, 53, 51, 43, 59, 58, 29, 71,
                          17, 22, 72, 18, 39, 35, 15, 38, 64, 52, 73, 67, 62, 50, 10, 68,
                          45, 63, 16, 14, 55, 54, 37, 20, 36, 12, 70, 34, 19, 26, 32, 23};

    vector<int> expected{//delta
                         0, 4, 0, 0, 0, 0, 0, 8, 0, 0, 8, 0, 0, 0, 0, 0, 0, 4, 4,  4, 12, 0,
                         0, 0, 0, 8, 0, 0, 4, 8, 0, 8, 0, 0, 8, 0, 0, 0, 0, 4, 16, 4, 16, 8,
                         0, 0, 0, 4, 0, 4, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0};

    copy_data(ep, dataEp);
    copy_data(input, dataInput);

    auto C = make_shared<op::Parameter>(element::i32, maxpool_shape);
    auto df = autodiff::backprop_function(f);
    backend->call(df, {output}, {input, ep});
    ASSERT_TRUE(read_vector<int>(output) == expected);
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_maxpool_n2_c1_hw5_3x3_str2_max)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape_a{1, 5, 5, 2}; //in CHWN
    Shape maxpool_shape{1, 2, 2, 2};

    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    auto reshape = make_shared<op::Reshape>(
        A, AxisVector{0, 3, 1, 2}, Shape{1, 2, 5, 5}); //convert CHWN to CNHW
    Shape window_shape{3, 3};
    auto window_movement_strides = Strides{2, 2};
    auto maxpool = make_shared<op::MaxPool>(reshape, window_shape, window_movement_strides);
    auto f = make_shared<Function>(maxpool, op::ParameterVector{A});

    shared_ptr<runtime::TensorView> ep = backend->create_tensor(element::i32, maxpool_shape);
    vector<int> dataEp(shape_size(maxpool_shape), 4);

    shared_ptr<runtime::TensorView> input = backend->create_tensor(element::i32, shape_a);
    shared_ptr<runtime::TensorView> output = backend->create_tensor(element::i32, shape_a);

    vector<int> dataInput{58, 15, 51, 35, 18, 47, 31, 32, 52, 21, 36, 38, 57, 54, 25, 45, 23,
                          30, 16, 27, 48, 20, 41, 37, 43, 39, 22, 28, 33, 29, 12, 17, 44, 42,
                          19, 40, 10, 46, 34, 53, 26, 55, 50, 13, 24, 14, 49, 56, 59, 11};

    vector<int> expected{//delta
                         4, 0, 0, 0, 0, 4, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 4, 4, 0};

    copy_data(ep, dataEp);
    copy_data(input, dataInput);

    auto C = make_shared<op::Parameter>(element::i32, maxpool_shape);
    auto df = autodiff::backprop_function(f);
    backend->call(df, {output}, {input, ep});
    ASSERT_TRUE(read_vector<int>(output) == expected);
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_avgpool_n1_c1_hw2x2)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape padding{1, 1};

    Shape shape_a{1, 1, 2, 2};
    Shape avgpool_shape{1, 1, 2, 2};

    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape window_shape{2, 2};
    auto window_movement_strides = Strides{2, 2};
    auto avgpool =
        make_shared<op::AvgPool>(A, window_shape, window_movement_strides, padding, padding, false);
    auto f = make_shared<Function>(avgpool, op::ParameterVector{A});

    shared_ptr<runtime::TensorView> ep = backend->create_tensor(element::i32, avgpool_shape);
    vector<int> dataEp(shape_size(avgpool_shape), 4);

    shared_ptr<runtime::TensorView> input = backend->create_tensor(element::i32, shape_a);

    shared_ptr<runtime::TensorView> output = backend->create_tensor(element::i32, shape_a);

    vector<int> dataInput{4, 8, 12, 16};

    vector<int> expected{1, 2, 3, 4};

    copy_data(ep, dataEp);
    copy_data(input, dataInput);

    auto C = make_shared<op::Parameter>(element::i32, avgpool_shape);
    auto df = autodiff::backprop_function(f);
    backend->call(df, {output}, {input, ep});
    ASSERT_TRUE(read_vector<int>(output) == dataEp);
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_avgpool_n1_c1_hw4x4)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape_a{1, 1, 4, 4};
    Shape avgpool_shape{1, 1, 3, 3};

    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape window_shape{2, 2};
    auto window_movement_strides = Strides{1, 1};
    auto avgpool = make_shared<op::AvgPool>(A, window_shape, window_movement_strides);
    auto f = make_shared<Function>(avgpool, op::ParameterVector{A});

    shared_ptr<runtime::TensorView> ep = backend->create_tensor(element::i32, avgpool_shape);
    vector<int> dataEp(shape_size(avgpool_shape), 4);

    shared_ptr<runtime::TensorView> input = backend->create_tensor(element::i32, shape_a);

    shared_ptr<runtime::TensorView> output = backend->create_tensor(element::i32, shape_a);

    vector<int> dataInput{1, 3, 1, 3, 1, 3, 1, 3, 3, 5, 3, 5, 3, 5, 3, 5};

    vector<int> expected{1, 2, 2, 1, 2, 4, 4, 2, 2, 4, 4, 2, 1, 2, 2, 1};

    copy_data(ep, dataEp);
    copy_data(input, dataInput);

    auto C = make_shared<op::Parameter>(element::i32, avgpool_shape);
    auto df = autodiff::backprop_function(f);
    backend->call(df, {output}, {input, ep});
    ASSERT_TRUE(read_vector<int>(output) == expected);
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_avgpool_n2_c2_hw4x4)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape_a{2, 2, 4, 4};
    Shape avgpool_shape{2, 2, 2, 2};

    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape window_shape{2, 2};
    auto window_movement_strides = Strides{2, 2};
    auto avgpool = make_shared<op::AvgPool>(A, window_shape, window_movement_strides);
    auto f = make_shared<Function>(avgpool, op::ParameterVector{A});

    shared_ptr<runtime::TensorView> ep = backend->create_tensor(element::i32, avgpool_shape);
    vector<int> dataEp(shape_size(avgpool_shape), 12);

    shared_ptr<runtime::TensorView> input = backend->create_tensor(element::i32, shape_a);

    shared_ptr<runtime::TensorView> output = backend->create_tensor(element::i32, shape_a);

    vector<int> dataInput{//i1c1
                          1,
                          2,
                          6,
                          7,
                          3,
                          4,
                          4,
                          3,
                          19,
                          1,
                          2,
                          3,
                          18,
                          2,
                          3,
                          2,
                          //i1c2
                          4,
                          1,
                          5,
                          5,
                          1,
                          4,
                          5,
                          5,
                          12,
                          8,
                          2,
                          3,
                          15,
                          5,
                          3,
                          2,
                          //i2c1
                          2,
                          3,
                          7,
                          7,
                          3,
                          2,
                          3,
                          3,
                          13,
                          7,
                          1,
                          2,
                          7,
                          13,
                          3,
                          4,
                          //i2c2
                          1,
                          1,
                          2,
                          2,
                          7,
                          1,
                          2,
                          14,
                          6,
                          16,
                          4,
                          1,
                          14,
                          4,
                          4,
                          1};

    vector<int> expected(shape_size(shape_a), 3);
    copy_data(ep, dataEp);
    copy_data(input, dataInput);

    auto C = make_shared<op::Parameter>(element::i32, avgpool_shape);
    auto df = autodiff::backprop_function(f);
    backend->call(df, {output}, {input, ep});
    ASSERT_TRUE(read_vector<int>(output) == expected);
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_avgpool_n2_c2_hw4x4_numeric)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    Shape shape_a{2, 2, 4, 4};
    test::Uniform<float> rng(1.0f, 10.0f);

    auto make_graph = [shape_a]() {
        auto A = make_shared<op::Parameter>(element::f32, shape_a);
        Shape window_shape{2, 2};
        auto window_movement_strides = Strides{2, 2};
        auto avgpool = make_shared<op::AvgPool>(A, window_shape, window_movement_strides);
        return make_shared<Function>(avgpool, op::ParameterVector{A});

    };

    for (auto i = 0; i < ${TEST_LOOPS}; i++)
    {
        auto x = rng.initialize(backend->create_tensor(element::f32, shape_a));
        EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x}, .01f, .01f));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_avgpool_n2_c2_hw4x4_win_2x2_str_1x1_numeric)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    Shape shape_a{2, 2, 4, 4};
    test::Uniform<float> rng(1.0f, 10.0f);

    auto make_graph = [shape_a]() {
        auto A = make_shared<op::Parameter>(element::f32, shape_a);
        Shape window_shape{2, 2};
        auto window_movement_strides = Strides{1, 1};
        auto avgpool = make_shared<op::AvgPool>(A, window_shape, window_movement_strides);
        return make_shared<Function>(avgpool, op::ParameterVector{A});

    };

    for (auto i = 0; i < ${TEST_LOOPS}; i++)
    {
        auto x = rng.initialize(backend->create_tensor(element::f32, shape_a));
        EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x}, .01f, .01f));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_avgpool_n2_c2_hw2x2_win_2x2_str_1x1_padding_numeric)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    Shape shape_a{2, 2, 4, 4};
    test::Uniform<float> rng(1.0f, 10.0f);

    auto make_graph = [shape_a]() {
        auto A = make_shared<op::Parameter>(element::f32, shape_a);
        Shape window_shape{2, 2};
        Shape padding{1, 1};
        auto window_movement_strides = Strides{2, 2};
        auto avgpool = make_shared<op::AvgPool>(
            A, window_shape, window_movement_strides, padding, padding, false);
        return make_shared<Function>(avgpool, op::ParameterVector{A});

    };

    for (auto i = 0; i < ${TEST_LOOPS}; i++)
    {
        auto x = rng.initialize(backend->create_tensor(element::f32, shape_a));
        EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x}, .01f, .01f));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_abs)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // The numeric derivative and the symbolic one may disagree around 0, so we will dance around
    // that point by skipping (-0.01,0.01).
    test::Uniform<float> rng_neg(-1.0f, -0.01f);
    test::Uniform<float> rng_pos(0.01f, 1.0f);
    Shape shape{2, 3};

    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Abs>(X),
                                     std::vector<std::shared_ptr<op::Parameter>>{X});
    };

    for (auto i = 0; i < ${TEST_LOOPS}; i++)
    {
        auto x_neg = rng_neg.initialize(backend->create_tensor<float>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x_neg}, .01f, .01f));

        auto x_pos = rng_pos.initialize(backend->create_tensor<float>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x_pos}, .01f, .01f));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_acos)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-0.9f, 0.9f);
    Shape shape{2, 3};
    auto x0 = rng.initialize(backend->create_tensor<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Acos>(X0),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_add)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-1.0f, 1.0f);
    Shape shape{2, 3};
    auto x0 = rng.initialize(backend->create_tensor<float>(shape));
    auto x1 = rng.initialize(backend->create_tensor<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        auto X1 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(X0 + X1, std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0, x1}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_add_nested)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-1.0f, 1.0f);
    Shape shape{2, 3};
    auto x0 = rng.initialize(backend->create_tensor<float>(shape));
    auto x1 = rng.initialize(backend->create_tensor<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        auto X1 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>((X0 + X1) + (X1 + X0),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0, x1}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_asin)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-0.9f, 0.9f);
    Shape shape{2, 3};
    auto x0 = rng.initialize(backend->create_tensor<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Asin>(X0),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_atan)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-10.0f, 10.0f);
    Shape shape{2, 3};
    auto x0 = rng.initialize(backend->create_tensor<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Atan>(X0),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_broadcast0)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-1.0f, 1.0f);
    Shape shape{3};
    auto x0 = rng.initialize(backend->create_tensor<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Broadcast>(X0, Shape{2, 3}, AxisSet{0}),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_broadcast1)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-1.0f, 1.0f);
    Shape shape{3};
    auto x0 = rng.initialize(backend->create_tensor<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Broadcast>(X0, Shape{3, 2}, AxisSet{1}),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_concat_vector)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-1.0f, 1.0f);
    Shape shape_0{3};
    auto x0 = rng.initialize(backend->create_tensor(element::f32, shape_0));
    Shape shape_1{2};
    auto x1 = rng.initialize(backend->create_tensor(element::f32, shape_1));
    Shape shape_2{1};
    auto x2 = rng.initialize(backend->create_tensor(element::f32, shape_2));

    auto make_graph = [shape_0, shape_1, shape_2]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape_0);
        auto X1 = make_shared<op::Parameter>(element::f32, shape_1);
        auto X2 = make_shared<op::Parameter>(element::f32, shape_2);
        return make_shared<Function>(make_shared<op::Concat>(NodeVector{X0, X1, X2}, 0),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1, X2});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0, x1, x2}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_concat_axis_0)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-1.0f, 1.0f);
    Shape shape_0{3, 2};
    auto x0 = rng.initialize(backend->create_tensor(element::f32, shape_0));
    Shape shape_1{2, 2};
    auto x1 = rng.initialize(backend->create_tensor(element::f32, shape_1));
    Shape shape_2{1, 2};
    auto x2 = rng.initialize(backend->create_tensor(element::f32, shape_2));

    auto make_graph = [shape_0, shape_1, shape_2]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape_0);
        auto X1 = make_shared<op::Parameter>(element::f32, shape_1);
        auto X2 = make_shared<op::Parameter>(element::f32, shape_2);
        return make_shared<Function>(make_shared<op::Concat>(NodeVector{X0, X1, X2}, 0),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1, X2});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0, x1, x2}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_concat_axis_1)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-1.0f, 1.0f);
    Shape shape_0{2, 3};
    auto x0 = rng.initialize(backend->create_tensor(element::f32, shape_0));
    Shape shape_1{2, 2};
    auto x1 = rng.initialize(backend->create_tensor(element::f32, shape_1));
    Shape shape_2{2, 1};
    auto x2 = rng.initialize(backend->create_tensor(element::f32, shape_2));

    auto make_graph = [shape_0, shape_1, shape_2]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape_0);
        auto X1 = make_shared<op::Parameter>(element::f32, shape_1);
        auto X2 = make_shared<op::Parameter>(element::f32, shape_2);
        return make_shared<Function>(make_shared<op::Concat>(NodeVector{X0, X1, X2}, 1),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1, X2});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0, x1, x2}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_ceiling)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // The numeric derivative and the symbolic one may disagree near integers, so we will dance around
    // them.
    test::Uniform<float> rng_minusone(-0.95f, -0.05f);
    test::Uniform<float> rng_plusone(0.05f, 0.95f);
    test::Uniform<float> rng_plustwo(1.05f, 1.95f);
    Shape shape{2, 3};

    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Ceiling>(X),
                                     std::vector<std::shared_ptr<op::Parameter>>{X});
    };

    for (auto i = 0; i < ${TEST_LOOPS}; i++)
    {
        auto x_minusone = rng_minusone.initialize(backend->create_tensor<float>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x_minusone}, .01f, .01f));

        auto x_plusone = rng_plusone.initialize(backend->create_tensor<float>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x_plusone}, .01f, .01f));

        auto x_plustwo = rng_plustwo.initialize(backend->create_tensor<float>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x_plustwo}, .01f, .01f));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_cos)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-10.0f, 10.0f);
    Shape shape{2, 3};
    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Cos>(X),
                                     std::vector<std::shared_ptr<op::Parameter>>{X});
    };

    for (auto i = 0; i < ${TEST_LOOPS}; i++)
    {
        auto x = rng.initialize(backend->create_tensor<float>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x}, .01f, .01f));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_cosh)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-10.0f, 10.0f);
    Shape shape{2, 3};
    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Cosh>(X),
                                     std::vector<std::shared_ptr<op::Parameter>>{X});
    };

    for (auto i = 0; i < ${TEST_LOOPS}; i++)
    {
        auto x = rng.initialize(backend->create_tensor<float>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x}, .01f, .01f));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_divide)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-1.0f, 1.0f);
    test::Uniform<float> rng1(1.0f, 2.0f);
    test::Uniform<float> rng2(-2.0f, -1.0f);
    Shape shape{2, 3};
    auto x0 = rng.initialize(backend->create_tensor<float>(shape));
    auto x1 = rng1.initialize(backend->create_tensor<float>(shape));
    auto x2 = rng2.initialize(backend->create_tensor<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        auto X1 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(X0 / X1, std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0, x1}, .01f, .01f));
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0, x2}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_dot_scalar_scalar)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-1.0f, 1.0f);
    Shape shape0{};
    Shape shape1{};
    auto x0 = rng.initialize(backend->create_tensor<float>(shape0));
    auto x1 = rng.initialize(backend->create_tensor<float>(shape1));

    auto make_graph = [shape0, shape1]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape0);
        auto X1 = make_shared<op::Parameter>(element::f32, shape1);
        return make_shared<Function>(make_shared<op::Dot>(X0, X1),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0, x1}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_dot_scalar_tensor)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-1.0f, 1.0f);
    Shape shape0{};
    Shape shape1{3, 4};
    auto x0 = rng.initialize(backend->create_tensor<float>(shape0));
    auto x1 = rng.initialize(backend->create_tensor<float>(shape1));

    auto make_graph = [shape0, shape1]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape0);
        auto X1 = make_shared<op::Parameter>(element::f32, shape1);
        return make_shared<Function>(make_shared<op::Dot>(X0, X1),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0, x1}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_dot_tensor_scalar)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-1.0f, 1.0f);
    Shape shape0{3, 4};
    Shape shape1{};
    auto x0 = rng.initialize(backend->create_tensor<float>(shape0));
    auto x1 = rng.initialize(backend->create_tensor<float>(shape1));

    auto make_graph = [shape0, shape1]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape0);
        auto X1 = make_shared<op::Parameter>(element::f32, shape1);
        return make_shared<Function>(make_shared<op::Dot>(X0, X1),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0, x1}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_dot_vector_vector)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-1.0f, 1.0f);
    Shape shape0{3};
    Shape shape1{3};
    auto x0 = rng.initialize(backend->create_tensor<float>(shape0));
    auto x1 = rng.initialize(backend->create_tensor<float>(shape1));

    auto make_graph = [shape0, shape1]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape0);
        auto X1 = make_shared<op::Parameter>(element::f32, shape1);
        return make_shared<Function>(make_shared<op::Dot>(X0, X1),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0, x1}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_dot_tensor_vector)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-1.0f, 1.0f);
    Shape shape0{4, 3};
    Shape shape1{3};
    auto x0 = rng.initialize(backend->create_tensor<float>(shape0));
    auto x1 = rng.initialize(backend->create_tensor<float>(shape1));

    auto make_graph = [shape0, shape1]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape0);
        auto X1 = make_shared<op::Parameter>(element::f32, shape1);
        return make_shared<Function>(make_shared<op::Dot>(X0, X1),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0, x1}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_dot_tensor2_tensor2)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-1.0f, 1.0f);
    Shape shape0{4, 3};
    Shape shape1{3, 5};
    auto x0 = rng.initialize(backend->create_tensor<float>(shape0));
    auto x1 = rng.initialize(backend->create_tensor<float>(shape1));

    auto make_graph = [shape0, shape1]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape0);
        auto X1 = make_shared<op::Parameter>(element::f32, shape1);
        return make_shared<Function>(make_shared<op::Dot>(X0, X1),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0, x1}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_dot_tensor3_tensor3)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-1.0f, 1.0f);
    Shape shape0{2, 4, 3};
    Shape shape1{4, 3, 3};
    auto x0 = rng.initialize(backend->create_tensor<float>(shape0));
    auto x1 = rng.initialize(backend->create_tensor<float>(shape1));

    auto make_graph = [shape0, shape1]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape0);
        auto X1 = make_shared<op::Parameter>(element::f32, shape1);
        return make_shared<Function>(make_shared<op::Dot>(X0, X1, 2),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0, x1}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_exp)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-1.0f, 1.0f);
    Shape shape{2, 3};
    auto x0 = rng.initialize(backend->create_tensor<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Exp>(X0),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_floor)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // The numeric derivative and the symbolic one may disagree near integers, so we will dance around
    // them.
    test::Uniform<float> rng_minusone(-0.95f, -0.05f);
    test::Uniform<float> rng_plusone(0.05f, 0.95f);
    test::Uniform<float> rng_plustwo(1.05f, 1.95f);
    Shape shape{2, 3};

    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Floor>(X),
                                     std::vector<std::shared_ptr<op::Parameter>>{X});
    };

    for (auto i = 0; i < ${TEST_LOOPS}; i++)
    {
        auto x_minusone = rng_minusone.initialize(backend->create_tensor<float>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x_minusone}, .01f, .01f));

        auto x_plusone = rng_plusone.initialize(backend->create_tensor<float>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x_plusone}, .01f, .01f));

        auto x_plustwo = rng_plustwo.initialize(backend->create_tensor<float>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x_plustwo}, .01f, .01f));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_log)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(1.0f, 2.0f);
    Shape shape{2, 3};
    auto x0 = rng.initialize(backend->create_tensor<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Log>(X0),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_maximum)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-1.0f, 1.0f);
    Shape shape{2, 3};
    auto x0 = rng.initialize(backend->create_tensor<float>(shape));
    auto x1 = rng.initialize(backend->create_tensor<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        auto X1 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Maximum>(X0, X1),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0, x1}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_minimum)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-1.0f, 1.0f);
    Shape shape{2, 3};
    auto x0 = rng.initialize(backend->create_tensor<float>(shape));
    auto x1 = rng.initialize(backend->create_tensor<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        auto X1 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Minimum>(X0, X1),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0, x1}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_multiply)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-1.0f, 1.0f);
    Shape shape{2, 3};
    auto x0 = rng.initialize(backend->create_tensor<float>(shape));
    auto x1 = rng.initialize(backend->create_tensor<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        auto X1 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(X0 * X1, std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0, x1}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_negative)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-1.0f, 1.0f);
    Shape shape{2, 3};
    auto x0 = rng.initialize(backend->create_tensor<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(-X0, std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_parameter)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-1.0f, 1.0f);
    Shape shape{2, 3};
    auto x0 = rng.initialize(backend->create_tensor<float>(shape));
    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(X0, std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_power)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng_neg(-5.0f, -0.5f);
    test::Uniform<float> rng_pos(0.5f, 5.0f);
    Shape shape{2, 3};

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        auto X1 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(std::make_shared<op::Power>(X0, X1),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };

    auto x0 = rng_neg.initialize(backend->create_tensor<float>(shape));
    auto x1 = rng_pos.initialize(backend->create_tensor<float>(shape));

    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0, x1}, .01f, .01f));

    x0 = rng_pos.initialize(backend->create_tensor<float>(shape));
    x1 = rng_neg.initialize(backend->create_tensor<float>(shape));

    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0, x1}, .01f, .01f));

    x0 = rng_pos.initialize(backend->create_tensor<float>(shape));
    x1 = rng_pos.initialize(backend->create_tensor<float>(shape));

    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0, x1}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_relu)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng_neg(-1.0f, -0.01f);
    test::Uniform<float> rng_pos(0.01f, 1.0f);
    Shape shape{2, 3};
    auto x0 = rng_neg.initialize(backend->create_tensor<float>(shape));
    auto x1 = rng_pos.initialize(backend->create_tensor<float>(shape));

    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Relu>(X),
                                     std::vector<std::shared_ptr<op::Parameter>>{X});
    };

    for (auto i = 0; i < ${TEST_LOOPS}; i++)
    {
        auto x_neg = rng_neg.initialize(backend->create_tensor<float>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x_neg}, .01f, .01f));

        auto x_pos = rng_pos.initialize(backend->create_tensor<float>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x_pos}, .01f, .01f));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_replace_slice)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-10.0f, 10.0f);
    Shape shape_x{5, 5};
    Shape shape_y{2, 2};
    auto make_graph = [shape_x, shape_y]() {
        auto X = make_shared<op::Parameter>(element::f32, shape_x);
        auto Y = make_shared<op::Parameter>(element::f32, shape_y);
        return make_shared<Function>(
            make_shared<op::ReplaceSlice>(X, Y, Coordinate{2, 3}, Coordinate{4, 5}),
            std::vector<std::shared_ptr<op::Parameter>>{X, Y});
    };

    for (auto i = 0; i < ${TEST_LOOPS}; i++)
    {
        auto x = rng.initialize(backend->create_tensor<float>(shape_x));
        auto y = rng.initialize(backend->create_tensor<float>(shape_y));

        EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x, y}, .01f, .01f));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_reshape)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-1.0f, 1.0f);
    Shape shape{3, 4};
    auto x0 = rng.initialize(backend->create_tensor<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Reshape>(X0, AxisVector{1, 0}, Shape{4, 3}),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_select)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-10.0f, 10.0f);
    Shape shape{2, 3};
    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::boolean, shape);
        auto X1 = make_shared<op::Parameter>(element::f32, shape);
        auto X2 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Select>(X0, X1, X2),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1, X2});
    };

    for (auto i = 0; i < ${TEST_LOOPS}; i++)
    {
        auto x0 = backend->create_tensor(element::boolean, shape);
        write_vector(x0, vector<char>{0, 1, 0, 1, 0, 1});
        auto x1 = rng.initialize(backend->create_tensor<float>(shape));
        auto x2 = rng.initialize(backend->create_tensor<float>(shape));

        EXPECT_TRUE(autodiff_numeric_compare_selective<float>(
            backend, make_graph, {x0, x1, x2}, .01f, .01f, std::vector<bool>{false, true, true}));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_select_nested)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-10.0f, 10.0f);
    Shape shape{2, 3};
    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::boolean, shape);
        auto X1 = make_shared<op::Parameter>(element::f32, shape);
        auto X2 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Select>(X0, X1 + X2, X2 - X1),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1, X2});
    };

    for (auto i = 0; i < ${TEST_LOOPS}; i++)
    {
        auto x0 = backend->create_tensor(element::boolean, shape);
        write_vector(x0, vector<char>{0, 1, 0, 1, 0, 1});
        auto x1 = rng.initialize(backend->create_tensor<float>(shape));
        auto x2 = rng.initialize(backend->create_tensor<float>(shape));

        EXPECT_TRUE(autodiff_numeric_compare_selective<float>(
            backend, make_graph, {x0, x1, x2}, .01f, .01f, std::vector<bool>{false, true, true}));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_sign)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // The numeric derivative and the symbolic one may disagree around 0, so we will dance around
    // that point by skipping (-0.01,0.01).
    test::Uniform<float> rng_neg(-1.0f, -0.01f);
    test::Uniform<float> rng_pos(0.01f, 1.0f);
    Shape shape{2, 3};

    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Sign>(X),
                                     std::vector<std::shared_ptr<op::Parameter>>{X});
    };

    for (auto i = 0; i < ${TEST_LOOPS}; i++)
    {
        auto x_neg = rng_neg.initialize(backend->create_tensor<float>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x_neg}, .01f, .01f));

        auto x_pos = rng_pos.initialize(backend->create_tensor<float>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x_pos}, .01f, .01f));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_sin)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-10.0f, 10.0f);
    Shape shape{2, 3};
    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Sin>(X),
                                     std::vector<std::shared_ptr<op::Parameter>>{X});
    };

    for (auto i = 0; i < ${TEST_LOOPS}; i++)
    {
        auto x = rng.initialize(backend->create_tensor<float>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x}, .01f, .01f));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_sinh)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-10.0f, 10.0f);
    Shape shape{2, 3};
    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Sinh>(X),
                                     std::vector<std::shared_ptr<op::Parameter>>{X});
    };

    for (auto i = 0; i < ${TEST_LOOPS}; i++)
    {
        auto x = rng.initialize(backend->create_tensor<float>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x}, .01f, .01f));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_slice)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    test::Uniform<float> rng(-10.0f, 10.0f);
    Shape shape{5, 5};
    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Slice>(X, Coordinate{2, 3}, Coordinate{4, 5}),
                                     std::vector<std::shared_ptr<op::Parameter>>{X});
    };

    for (auto i = 0; i < ${TEST_LOOPS}; i++)
    {
        auto x = rng.initialize(backend->create_tensor<float>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x}, .01f, .01f));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_softmax_all)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-1.0f, 1.0f);
    Shape shape{2, 3};
    auto x0 = rng.initialize(backend->create_tensor<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Softmax>(X0, AxisSet{0, 1}),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_softmax_axis)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-1.0f, 1.0f);
    Shape shape{2, 3};
    auto x0 = rng.initialize(backend->create_tensor<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Softmax>(X0, AxisSet{1}),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_softmax_underflow)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto low = std::numeric_limits<float>::lowest();

    Shape shape{2, 3};
    auto x0 = backend->create_tensor(element::f32, shape);
    copy_data(x0, vector<float>{low, 1, 2, 3, 4, 5});

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Softmax>(X0, AxisSet{0, 1}),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_softmax_3d)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-1.0f, 1.0f);
    Shape shape{2, 3, 4};
    auto x0 = rng.initialize(backend->create_tensor<float>(shape));

    auto make_graph0 = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Softmax>(X0, AxisSet{0}),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph0, {x0}, .01f, .01f));

    auto make_graph1 = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Softmax>(X0, AxisSet{1}),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph1, {x0}, .01f, .01f));

    auto make_graph2 = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Softmax>(X0, AxisSet{2}),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph2, {x0}, .01f, .01f));

    auto make_graph01 = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Softmax>(X0, AxisSet{0, 1}),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph01, {x0}, .01f, .01f));

    auto make_graph02 = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Softmax>(X0, AxisSet{0, 2}),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph02, {x0}, .01f, .01f));

    auto make_graph12 = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Softmax>(X0, AxisSet{1, 2}),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph12, {x0}, .01f, .01f));

    auto make_graph012 = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Softmax>(X0, AxisSet{0, 1, 2}),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph012, {x0}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_subtract)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-1.0f, 1.0f);
    Shape shape{2, 3};
    auto x0 = rng.initialize(backend->create_tensor<float>(shape));
    auto x1 = rng.initialize(backend->create_tensor<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        auto X1 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(X0 - X1, std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0, x1}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_sum_v2s)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-1.0f, 1.0f);
    Shape shape{8};
    auto x = rng.initialize(backend->create_tensor<float>(shape));

    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Sum>(X, AxisSet{0}),
                                     std::vector<std::shared_ptr<op::Parameter>>{X});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_sum_m2s)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-1.0f, 1.0f);
    Shape shape{8, 9};
    auto x = rng.initialize(backend->create_tensor<float>(shape));

    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Sum>(X, AxisSet{0, 1}),
                                     std::vector<std::shared_ptr<op::Parameter>>{X});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_sum_m2v_0)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-1.0f, 1.0f);
    Shape shape{8, 9};
    auto x = rng.initialize(backend->create_tensor<float>(shape));

    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Sum>(X, AxisSet{0}),
                                     std::vector<std::shared_ptr<op::Parameter>>{X});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_sum_m2v_1)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-1.0f, 1.0f);
    Shape shape{8, 9};
    auto x = rng.initialize(backend->create_tensor<float>(shape));

    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Sum>(X, AxisSet{1}),
                                     std::vector<std::shared_ptr<op::Parameter>>{X});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_tan)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    auto pi = 3.14159f;

    // Stay away from the asymptotes at 6 and 12 o'clock.
    auto slop = 0.1f;
    test::Uniform<float> rng_r(-pi / 2 + slop, pi / 2 - slop);
    test::Uniform<float> rng_l(pi / 2 + slop, (3 * pi) / 2 - slop);

    Shape shape{2, 3};

    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Tan>(X),
                                     std::vector<std::shared_ptr<op::Parameter>>{X});
    };

    for (auto i = 0; i < ${TEST_LOOPS}; i++)
    {
        auto x_r = rng_r.initialize(backend->create_tensor<float>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x_r}, .01f, .01f));

        auto x_l = rng_l.initialize(backend->create_tensor<float>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x_l}, .01f, .01f));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_tanh)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-10.0f, 10.0f);
    Shape shape{2, 3};
    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Tanh>(X),
                                     std::vector<std::shared_ptr<op::Parameter>>{X});
    };

    for (auto i = 0; i < ${TEST_LOOPS}; i++)
    {
        auto x = rng.initialize(backend->create_tensor<float>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x}, .01f, .01f));
    }
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_abc)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-1.0f, 1.0f);
    Shape shape{2, 3};
    auto x0 = rng.initialize(backend->create_tensor<float>(shape));
    auto x1 = rng.initialize(backend->create_tensor<float>(shape));
    auto x2 = rng.initialize(backend->create_tensor<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        auto X1 = make_shared<op::Parameter>(element::f32, shape);
        auto X2 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>((X0 + X1) * X2,
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1, X2});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x0, x1, x2}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_reverse_3d_02)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    test::Uniform<float> rng(-1.0f, 1.0f);
    Shape shape{2, 4, 5};
    auto x = rng.initialize(backend->create_tensor(element::f32, shape));

    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Reverse>(X, AxisSet{0, 2}),
                                     std::vector<std::shared_ptr<op::Parameter>>{X});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(backend, make_graph, {x}, .01f, .01f));
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_maxpool_n4c1h4w4_kh2kw2_sh1sw1)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");
    Shape shape_a{4, 1, 4, 4}; //in NCHW
    Shape maxpool_shape{4, 1, 3, 3};

    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape window_shape{2, 2};
    auto window_movement_strides = Strides{1, 1};
    auto maxpool = make_shared<op::MaxPool>(A, window_shape, window_movement_strides);
    auto f = make_shared<Function>(maxpool, op::ParameterVector{A});
    shared_ptr<runtime::TensorView> ep = backend->create_tensor(element::f32, maxpool_shape);
    vector<float> dataEp(shape_size(maxpool_shape), 4);

    shared_ptr<runtime::TensorView> input = backend->create_tensor(element::f32, shape_a);
    shared_ptr<runtime::TensorView> output = backend->create_tensor(element::f32, shape_a);

    vector<float> dataInput{11, 65, 44, 28, 31, 33, 21, 66, 40, 49, 69, 57, 47, 30, 24, 27,
                            13, 56, 46, 60, 61, 41, 25, 42, 48, 53, 51, 43, 59, 58, 29, 71,
                            17, 22, 72, 18, 39, 35, 15, 38, 64, 52, 73, 67, 62, 50, 10, 68,
                            45, 63, 16, 14, 55, 54, 37, 20, 36, 12, 70, 34, 19, 26, 32, 23};

    vector<float> expected{//delta
                           0, 8, 0, 0, 0, 0, 0, 4, 0, 8, 16, 0, 0, 0, 0,  0, 0, 4, 0, 4, 8,  0,
                           0, 0, 0, 4, 4, 0, 4, 4, 0, 4, 0,  0, 8, 0, 4,  0, 0, 0, 8, 0, 16, 0,
                           0, 0, 0, 0, 0, 8, 0, 0, 4, 0, 4,  0, 4, 0, 16, 0, 0, 0, 0, 0};

    copy_data(ep, dataEp);
    copy_data(input, dataInput);

    auto C = make_shared<op::Parameter>(element::f32, maxpool_shape);
    auto df = autodiff::backprop_function(f);
    backend->call(df, {output}, {input, ep});
    ASSERT_TRUE(read_vector<float>(output) == expected);
}

NGRAPH_TEST(${BACKEND_NAME}, backwards_maxpool_n2c1h5w5_kh3kw3_sh2sw2)
{
    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    Shape shape_a{1, 2, 5, 5}; //in NCHW
    Shape maxpool_shape{1, 2, 2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape window_shape{3, 3};
    auto window_movement_strides = Strides{2, 2};
    auto maxpool = make_shared<op::MaxPool>(A, window_shape, window_movement_strides);
    auto f = make_shared<Function>(maxpool, op::ParameterVector{A});

    shared_ptr<runtime::TensorView> ep = backend->create_tensor(element::f32, maxpool_shape);
    vector<float> dataEp(shape_size(maxpool_shape), 4);

    shared_ptr<runtime::TensorView> input = backend->create_tensor(element::f32, shape_a);
    shared_ptr<runtime::TensorView> output = backend->create_tensor(element::f32, shape_a);

    vector<float> dataInput{58, 15, 51, 35, 18, 47, 31, 32, 52, 21, 36, 38, 57, 54, 25, 45, 23,
                            30, 16, 27, 48, 20, 41, 37, 43, 39, 22, 28, 33, 29, 12, 17, 44, 42,
                            19, 40, 10, 46, 34, 53, 26, 55, 50, 13, 24, 14, 49, 56, 59, 11};

    vector<float> expected{//delta
                           4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0,
                           0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0,  4, 4, 0};

    copy_data(ep, dataEp);
    copy_data(input, dataInput);

    auto C = make_shared<op::Parameter>(element::f32, maxpool_shape);
    auto df = autodiff::backprop_function(f);
    backend->call(df, {output}, {input, ep});
    ASSERT_TRUE(read_vector<float>(output) == expected);
}
