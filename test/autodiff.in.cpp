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
#include <functional>
#include <memory>
#include <tuple>

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/autodiff/backprop_derivative.hpp"
#include "util/autodiff/backprop_function.hpp"
#include "util/autodiff/numeric_derivative.hpp"
#include "util/random.hpp"

using namespace std;
using namespace ngraph;

template <typename T>
bool autodiff_numeric_compare(const std::shared_ptr<runtime::Manager>& manager,
                              const std::shared_ptr<runtime::Backend>& backend,
                              std::function<std::shared_ptr<Function>()> make_graph,
                              const std::vector<std::shared_ptr<runtime::TensorView>>& args,
                              T rtol,
                              T atol)
{
    auto f = make_graph();
    auto results_num =
        autodiff::numeric_derivative<T>(manager, backend, f, args, .001f, f->get_parameters());

    auto g = make_graph();
    auto results_sym =
        autodiff::backprop_derivative<T>(manager, backend, g, args, g->get_parameters());

    return test::all_close(results_num, results_sym, rtol, atol);
}

template <typename T>
bool autodiff_numeric_compare_selective(
    const std::shared_ptr<runtime::Manager>& manager,
    const std::shared_ptr<runtime::Backend>& backend,
    std::function<std::shared_ptr<Function>()> make_graph,
    const std::vector<std::shared_ptr<runtime::TensorView>>& args,
    T rtol,
    T atol,
    const std::vector<bool>& indep_param_mask)
{
    std::vector<std::shared_ptr<op::Parameter>> f_indep_params;
    auto f = make_graph();

    size_t i = 0;

    for (auto b : indep_param_mask)
    {
        if (b)
        {
            f_indep_params.push_back(f->get_parameters().at(i));
        }
        i++;
    }

    auto results_num =
        autodiff::numeric_derivative<T>(manager, backend, f, args, .001f, f_indep_params);

    std::vector<std::shared_ptr<op::Parameter>> g_indep_params;
    auto g = make_graph();

    i = 0;

    for (auto b : indep_param_mask)
    {
        if (b)
        {
            g_indep_params.push_back(g->get_parameters().at(i));
        }
        i++;
    }

    auto results_sym = autodiff::backprop_derivative<T>(manager, backend, g, args, g_indep_params);

    return test::all_close(results_num, results_sym, rtol, atol);
}

TEST(${BACKEND_NAME}, backwards_maxpool_n4_c1_hw4_2x2_max)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    auto shape_a = Shape{1, 4, 4, 4}; //in CHWN
    auto maxpool_shape = Shape{1, 4, 3, 3};

    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    auto reshape = make_shared<op::Reshape>(
        A, AxisVector{0, 3, 1, 2}, Shape{1, 4, 4, 4}); //convert CHWN to CNHW
    auto window_shape = Shape{2, 2};
    auto window_movement_strides = Strides{1, 1};
    auto maxpool = make_shared<op::MaxPool>(reshape, window_shape, window_movement_strides);
    auto f = make_shared<Function>(maxpool, op::Parameters{A});

    shared_ptr<runtime::TensorView> ep =
        backend->make_primary_tensor_view(element::i32, maxpool_shape);
    vector<int> dataEp(shape_size(maxpool_shape), 4);

    shared_ptr<runtime::TensorView> input =
        backend->make_primary_tensor_view(element::i32, shape_a);
    shared_ptr<runtime::TensorView> output =
        backend->make_primary_tensor_view(element::i32, shape_a);

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
    auto external = manager->compile(df);
    auto cf = backend->make_call_frame(external);
    cf->tensor_call({input, ep}, {output});
    ASSERT_TRUE(get_vector<int>(output) == expected);
}

TEST(${BACKEND_NAME}, backwards_maxpool_n2_c1_hw5_3x3_str2_max)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    auto shape_a = Shape{1, 5, 5, 2}; //in CHWN
    auto maxpool_shape = Shape{1, 2, 2, 2};

    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    auto reshape = make_shared<op::Reshape>(
        A, AxisVector{0, 3, 1, 2}, Shape{1, 2, 5, 5}); //convert CHWN to CNHW
    auto window_shape = Shape{3, 3};
    auto window_movement_strides = Strides{2, 2};
    auto maxpool = make_shared<op::MaxPool>(reshape, window_shape, window_movement_strides);
    auto f = make_shared<Function>(maxpool, op::Parameters{A});

    shared_ptr<runtime::TensorView> ep =
        backend->make_primary_tensor_view(element::i32, maxpool_shape);
    vector<int> dataEp(shape_size(maxpool_shape), 4);

    shared_ptr<runtime::TensorView> input =
        backend->make_primary_tensor_view(element::i32, shape_a);
    shared_ptr<runtime::TensorView> output =
        backend->make_primary_tensor_view(element::i32, shape_a);

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
    auto external = manager->compile(df);
    auto cf = backend->make_call_frame(external);
    cf->tensor_call({input, ep}, {output});
    ASSERT_TRUE(get_vector<int>(output) == expected);
}

TEST(${BACKEND_NAME}, backwards_abs)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    // The numeric derivative and the symbolic one may disagree around 0, so we will dance around
    // that point by skipping (-0.01,0.01).
    test::Uniform<float> rng_neg(-1.0f, -0.01f);
    test::Uniform<float> rng_pos(0.01f, 1.0f);
    auto shape = Shape{2, 3};

    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Abs>(X),
                                     std::vector<std::shared_ptr<op::Parameter>>{X});
    };

    for (auto i = 0; i < 100; i++)
    {
        auto x_neg = rng_neg.initialize(backend->make_primary_tensor_view<float>(shape));

        EXPECT_TRUE(
            autodiff_numeric_compare<float>(manager, backend, make_graph, {x_neg}, .01f, .01f));

        auto x_pos = rng_pos.initialize(backend->make_primary_tensor_view<float>(shape));

        EXPECT_TRUE(
            autodiff_numeric_compare<float>(manager, backend, make_graph, {x_pos}, .01f, .01f));
    }
}

TEST(${BACKEND_NAME}, backwards_add)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape));
    auto x1 = rng.initialize(backend->make_primary_tensor_view<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        auto X1 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(X0 + X1, std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(${BACKEND_NAME}, backwards_add_nested)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape));
    auto x1 = rng.initialize(backend->make_primary_tensor_view<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        auto X1 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>((X0 + X1) + (X1 + X0),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(${BACKEND_NAME}, backwards_broadcast0)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape = Shape{3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Broadcast>(X0, Shape{2, 3}, AxisSet{0}),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(manager, backend, make_graph, {x0}, .01f, .01f));
}

TEST(${BACKEND_NAME}, backwards_broadcast1)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape = Shape{3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Broadcast>(X0, Shape{3, 2}, AxisSet{1}),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(manager, backend, make_graph, {x0}, .01f, .01f));
}

TEST(${BACKEND_NAME}, backwards_concat_vector)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape_0 = Shape{3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view(element::f32, shape_0));
    auto shape_1 = Shape{2};
    auto x1 = rng.initialize(backend->make_primary_tensor_view(element::f32, shape_1));
    auto shape_2 = Shape{1};
    auto x2 = rng.initialize(backend->make_primary_tensor_view(element::f32, shape_2));

    auto make_graph = [shape_0, shape_1, shape_2]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape_0);
        auto X1 = make_shared<op::Parameter>(element::f32, shape_1);
        auto X2 = make_shared<op::Parameter>(element::f32, shape_2);
        return make_shared<Function>(make_shared<op::Concat>(Nodes{X0, X1, X2}, 0),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1, X2});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1, x2}, .01f, .01f));
}

TEST(${BACKEND_NAME}, backwards_concat_axis_0)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape_0 = Shape{3, 2};
    auto x0 = rng.initialize(backend->make_primary_tensor_view(element::f32, shape_0));
    auto shape_1 = Shape{2, 2};
    auto x1 = rng.initialize(backend->make_primary_tensor_view(element::f32, shape_1));
    auto shape_2 = Shape{1, 2};
    auto x2 = rng.initialize(backend->make_primary_tensor_view(element::f32, shape_2));

    auto make_graph = [shape_0, shape_1, shape_2]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape_0);
        auto X1 = make_shared<op::Parameter>(element::f32, shape_1);
        auto X2 = make_shared<op::Parameter>(element::f32, shape_2);
        return make_shared<Function>(make_shared<op::Concat>(Nodes{X0, X1, X2}, 0),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1, X2});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1, x2}, .01f, .01f));
}

TEST(${BACKEND_NAME}, backwards_concat_axis_1)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape_0 = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view(element::f32, shape_0));
    auto shape_1 = Shape{2, 2};
    auto x1 = rng.initialize(backend->make_primary_tensor_view(element::f32, shape_1));
    auto shape_2 = Shape{2, 1};
    auto x2 = rng.initialize(backend->make_primary_tensor_view(element::f32, shape_2));

    auto make_graph = [shape_0, shape_1, shape_2]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape_0);
        auto X1 = make_shared<op::Parameter>(element::f32, shape_1);
        auto X2 = make_shared<op::Parameter>(element::f32, shape_2);
        return make_shared<Function>(make_shared<op::Concat>(Nodes{X0, X1, X2}, 1),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1, X2});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1, x2}, .01f, .01f));
}

TEST(${BACKEND_NAME}, backwards_ceiling)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    // The numeric derivative and the symbolic one may disagree near integers, so we will dance around
    // them.
    test::Uniform<float> rng_minusone(-0.95f, -0.05f);
    test::Uniform<float> rng_plusone(0.05f, 0.95f);
    test::Uniform<float> rng_plustwo(1.05f, 1.95f);
    auto shape = Shape{2, 3};

    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Ceiling>(X),
                                     std::vector<std::shared_ptr<op::Parameter>>{X});
    };

    for (auto i = 0; i < 100; i++)
    {
        auto x_minusone = rng_minusone.initialize(backend->make_primary_tensor_view<float>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<float>(
            manager, backend, make_graph, {x_minusone}, .01f, .01f));

        auto x_plusone = rng_plusone.initialize(backend->make_primary_tensor_view<float>(shape));

        EXPECT_TRUE(
            autodiff_numeric_compare<float>(manager, backend, make_graph, {x_plusone}, .01f, .01f));

        auto x_plustwo = rng_plustwo.initialize(backend->make_primary_tensor_view<float>(shape));

        EXPECT_TRUE(
            autodiff_numeric_compare<float>(manager, backend, make_graph, {x_plustwo}, .01f, .01f));
    }
}

TEST(${BACKEND_NAME}, backwards_cos)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-10.0f, 10.0f);
    auto shape = Shape{2, 3};
    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Cos>(X),
                                     std::vector<std::shared_ptr<op::Parameter>>{X});
    };

    for (auto i = 0; i < 100; i++)
    {
        auto x = rng.initialize(backend->make_primary_tensor_view<float>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<float>(manager, backend, make_graph, {x}, .01f, .01f));
    }
}

TEST(${BACKEND_NAME}, backwards_cosh)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-10.0f, 10.0f);
    auto shape = Shape{2, 3};
    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Cosh>(X),
                                     std::vector<std::shared_ptr<op::Parameter>>{X});
    };

    for (auto i = 0; i < 100; i++)
    {
        auto x = rng.initialize(backend->make_primary_tensor_view<float>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<float>(manager, backend, make_graph, {x}, .01f, .01f));
    }
}

TEST(${BACKEND_NAME}, backwards_divide)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    test::Uniform<float> rng1(1.0f, 2.0f);
    test::Uniform<float> rng2(-2.0f, -1.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape));
    auto x1 = rng1.initialize(backend->make_primary_tensor_view<float>(shape));
    auto x2 = rng2.initialize(backend->make_primary_tensor_view<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        auto X1 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(X0 / X1, std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1}, .01f, .01f));
    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x2}, .01f, .01f));
}

TEST(${BACKEND_NAME}, backwards_dot_scalar_scalar)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape0 = Shape{};
    auto shape1 = Shape{};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape0));
    auto x1 = rng.initialize(backend->make_primary_tensor_view<float>(shape1));

    auto make_graph = [shape0, shape1]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape0);
        auto X1 = make_shared<op::Parameter>(element::f32, shape1);
        return make_shared<Function>(make_shared<op::Dot>(X0, X1),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(${BACKEND_NAME}, backwards_dot_scalar_tensor)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape0 = Shape{};
    auto shape1 = Shape{3, 4};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape0));
    auto x1 = rng.initialize(backend->make_primary_tensor_view<float>(shape1));

    auto make_graph = [shape0, shape1]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape0);
        auto X1 = make_shared<op::Parameter>(element::f32, shape1);
        return make_shared<Function>(make_shared<op::Dot>(X0, X1),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(${BACKEND_NAME}, backwards_dot_tensor_scalar)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape0 = Shape{3, 4};
    auto shape1 = Shape{};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape0));
    auto x1 = rng.initialize(backend->make_primary_tensor_view<float>(shape1));

    auto make_graph = [shape0, shape1]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape0);
        auto X1 = make_shared<op::Parameter>(element::f32, shape1);
        return make_shared<Function>(make_shared<op::Dot>(X0, X1),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(${BACKEND_NAME}, backwards_dot_vector_vector)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape0 = Shape{3};
    auto shape1 = Shape{3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape0));
    auto x1 = rng.initialize(backend->make_primary_tensor_view<float>(shape1));

    auto make_graph = [shape0, shape1]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape0);
        auto X1 = make_shared<op::Parameter>(element::f32, shape1);
        return make_shared<Function>(make_shared<op::Dot>(X0, X1),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(${BACKEND_NAME}, backwards_dot_tensor_vector)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape0 = Shape{4, 3};
    auto shape1 = Shape{3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape0));
    auto x1 = rng.initialize(backend->make_primary_tensor_view<float>(shape1));

    auto make_graph = [shape0, shape1]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape0);
        auto X1 = make_shared<op::Parameter>(element::f32, shape1);
        return make_shared<Function>(make_shared<op::Dot>(X0, X1),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(${BACKEND_NAME}, backwards_dot_tensor2_tensor2)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape0 = Shape{4, 3};
    auto shape1 = Shape{3, 5};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape0));
    auto x1 = rng.initialize(backend->make_primary_tensor_view<float>(shape1));

    auto make_graph = [shape0, shape1]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape0);
        auto X1 = make_shared<op::Parameter>(element::f32, shape1);
        return make_shared<Function>(make_shared<op::Dot>(X0, X1),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(${BACKEND_NAME}, backwards_dot_tensor3_tensor3)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape0 = Shape{2, 4, 3};
    auto shape1 = Shape{4, 3, 3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape0));
    auto x1 = rng.initialize(backend->make_primary_tensor_view<float>(shape1));

    auto make_graph = [shape0, shape1]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape0);
        auto X1 = make_shared<op::Parameter>(element::f32, shape1);
        return make_shared<Function>(make_shared<op::Dot>(X0, X1, 2),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(${BACKEND_NAME}, backwards_exp)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Exp>(X0),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(manager, backend, make_graph, {x0}, .01f, .01f));
}

TEST(${BACKEND_NAME}, backwards_floor)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    // The numeric derivative and the symbolic one may disagree near integers, so we will dance around
    // them.
    test::Uniform<float> rng_minusone(-0.95f, -0.05f);
    test::Uniform<float> rng_plusone(0.05f, 0.95f);
    test::Uniform<float> rng_plustwo(1.05f, 1.95f);
    auto shape = Shape{2, 3};

    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Floor>(X),
                                     std::vector<std::shared_ptr<op::Parameter>>{X});
    };

    for (auto i = 0; i < 100; i++)
    {
        auto x_minusone = rng_minusone.initialize(backend->make_primary_tensor_view<float>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<float>(
            manager, backend, make_graph, {x_minusone}, .01f, .01f));

        auto x_plusone = rng_plusone.initialize(backend->make_primary_tensor_view<float>(shape));

        EXPECT_TRUE(
            autodiff_numeric_compare<float>(manager, backend, make_graph, {x_plusone}, .01f, .01f));

        auto x_plustwo = rng_plustwo.initialize(backend->make_primary_tensor_view<float>(shape));

        EXPECT_TRUE(
            autodiff_numeric_compare<float>(manager, backend, make_graph, {x_plustwo}, .01f, .01f));
    }
}

TEST(${BACKEND_NAME}, backwards_log)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(1.0f, 2.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Log>(X0),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(manager, backend, make_graph, {x0}, .01f, .01f));
}

TEST(${BACKEND_NAME}, backwards_maximum)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape));
    auto x1 = rng.initialize(backend->make_primary_tensor_view<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        auto X1 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Maximum>(X0, X1),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(${BACKEND_NAME}, backwards_minimum)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape));
    auto x1 = rng.initialize(backend->make_primary_tensor_view<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        auto X1 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Minimum>(X0, X1),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(${BACKEND_NAME}, backwards_multiply)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape));
    auto x1 = rng.initialize(backend->make_primary_tensor_view<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        auto X1 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(X0 * X1, std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(${BACKEND_NAME}, backwards_negative)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(-X0, std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(manager, backend, make_graph, {x0}, .01f, .01f));
}

TEST(${BACKEND_NAME}, backwards_parameter)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape));
    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(X0, std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(manager, backend, make_graph, {x0}, .01f, .01f));
}

TEST(${BACKEND_NAME}, backwards_power)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng_neg(-5.0f, -0.5f);
    test::Uniform<float> rng_pos(0.5f, 5.0f);
    auto shape = Shape{2, 3};

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        auto X1 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(std::make_shared<op::Power>(X0, X1),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };

    auto x0 = rng_neg.initialize(backend->make_primary_tensor_view<float>(shape));
    auto x1 = rng_pos.initialize(backend->make_primary_tensor_view<float>(shape));

    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1}, .01f, .01f));

    x0 = rng_pos.initialize(backend->make_primary_tensor_view<float>(shape));
    x1 = rng_neg.initialize(backend->make_primary_tensor_view<float>(shape));

    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1}, .01f, .01f));

    x0 = rng_neg.initialize(backend->make_primary_tensor_view<float>(shape));
    x1 = rng_neg.initialize(backend->make_primary_tensor_view<float>(shape));

    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1}, .01f, .01f));

    x0 = rng_pos.initialize(backend->make_primary_tensor_view<float>(shape));
    x1 = rng_pos.initialize(backend->make_primary_tensor_view<float>(shape));

    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(${BACKEND_NAME}, backwards_replace_slice)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-10.0f, 10.0f);
    auto shape_x = Shape{5, 5};
    auto shape_y = Shape{2, 2};
    auto make_graph = [shape_x, shape_y]() {
        auto X = make_shared<op::Parameter>(element::f32, shape_x);
        auto Y = make_shared<op::Parameter>(element::f32, shape_y);
        return make_shared<Function>(
            make_shared<op::ReplaceSlice>(X, Y, Coordinate{2, 3}, Coordinate{4, 5}),
            std::vector<std::shared_ptr<op::Parameter>>{X, Y});
    };

    for (auto i = 0; i < 100; i++)
    {
        auto x = rng.initialize(backend->make_primary_tensor_view<float>(shape_x));
        auto y = rng.initialize(backend->make_primary_tensor_view<float>(shape_y));

        EXPECT_TRUE(
            autodiff_numeric_compare<float>(manager, backend, make_graph, {x, y}, .01f, .01f));
    }
}

TEST(${BACKEND_NAME}, backwards_reshape)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape = Shape{3, 4};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Reshape>(X0, AxisVector{1, 0}, Shape{4, 3}),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(manager, backend, make_graph, {x0}, .01f, .01f));
}

TEST(${BACKEND_NAME}, backwards_select)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-10.0f, 10.0f);
    auto shape = Shape{2, 3};
    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::boolean, shape);
        auto X1 = make_shared<op::Parameter>(element::f32, shape);
        auto X2 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Select>(X0, X1, X2),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1, X2});
    };

    for (auto i = 0; i < 100; i++)
    {
        auto x0 = backend->make_primary_tensor_view(element::boolean, shape);
        set_vector(x0, vector<char>{0, 1, 0, 1, 0, 1});
        auto x1 = rng.initialize(backend->make_primary_tensor_view<float>(shape));
        auto x2 = rng.initialize(backend->make_primary_tensor_view<float>(shape));

        EXPECT_TRUE(
            autodiff_numeric_compare_selective<float>(manager,
                                                      backend,
                                                      make_graph,
                                                      {x0, x1, x2},
                                                      .01f,
                                                      .01f,
                                                      std::vector<bool>{false, true, true}));
    }
}

TEST(${BACKEND_NAME}, backwards_select_nested)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-10.0f, 10.0f);
    auto shape = Shape{2, 3};
    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::boolean, shape);
        auto X1 = make_shared<op::Parameter>(element::f32, shape);
        auto X2 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Select>(X0, X1 + X2, X2 - X1),
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1, X2});
    };

    for (auto i = 0; i < 100; i++)
    {
        auto x0 = backend->make_primary_tensor_view(element::boolean, shape);
        set_vector(x0, vector<char>{0, 1, 0, 1, 0, 1});
        auto x1 = rng.initialize(backend->make_primary_tensor_view<float>(shape));
        auto x2 = rng.initialize(backend->make_primary_tensor_view<float>(shape));

        EXPECT_TRUE(
            autodiff_numeric_compare_selective<float>(manager,
                                                      backend,
                                                      make_graph,
                                                      {x0, x1, x2},
                                                      .01f,
                                                      .01f,
                                                      std::vector<bool>{false, true, true}));
    }
}

TEST(${BACKEND_NAME}, backwards_sign)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    // The numeric derivative and the symbolic one may disagree around 0, so we will dance around
    // that point by skipping (-0.01,0.01).
    test::Uniform<float> rng_neg(-1.0f, -0.01f);
    test::Uniform<float> rng_pos(0.01f, 1.0f);
    auto shape = Shape{2, 3};

    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Sign>(X),
                                     std::vector<std::shared_ptr<op::Parameter>>{X});
    };

    for (auto i = 0; i < 100; i++)
    {
        auto x_neg = rng_neg.initialize(backend->make_primary_tensor_view<float>(shape));

        EXPECT_TRUE(
            autodiff_numeric_compare<float>(manager, backend, make_graph, {x_neg}, .01f, .01f));

        auto x_pos = rng_pos.initialize(backend->make_primary_tensor_view<float>(shape));

        EXPECT_TRUE(
            autodiff_numeric_compare<float>(manager, backend, make_graph, {x_pos}, .01f, .01f));
    }
}

TEST(${BACKEND_NAME}, backwards_sin)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-10.0f, 10.0f);
    auto shape = Shape{2, 3};
    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Sin>(X),
                                     std::vector<std::shared_ptr<op::Parameter>>{X});
    };

    for (auto i = 0; i < 100; i++)
    {
        auto x = rng.initialize(backend->make_primary_tensor_view<float>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<float>(manager, backend, make_graph, {x}, .01f, .01f));
    }
}

TEST(${BACKEND_NAME}, backwards_sinh)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-10.0f, 10.0f);
    auto shape = Shape{2, 3};
    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Sinh>(X),
                                     std::vector<std::shared_ptr<op::Parameter>>{X});
    };

    for (auto i = 0; i < 100; i++)
    {
        auto x = rng.initialize(backend->make_primary_tensor_view<float>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<float>(manager, backend, make_graph, {x}, .01f, .01f));
    }
}

TEST(${BACKEND_NAME}, backwards_slice)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-10.0f, 10.0f);
    auto shape = Shape{5, 5};
    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Slice>(X, Coordinate{2, 3}, Coordinate{4, 5}),
                                     std::vector<std::shared_ptr<op::Parameter>>{X});
    };

    for (auto i = 0; i < 100; i++)
    {
        auto x = rng.initialize(backend->make_primary_tensor_view<float>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<float>(manager, backend, make_graph, {x}, .01f, .01f));
    }
}

TEST(${BACKEND_NAME}, backwards_sqrt)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    // Deriv has an asymptote at 0 so we'll stay away from there.
    test::Uniform<float> rng(0.1f, 10.0f);
    auto shape = Shape{2, 3};
    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Sqrt>(X),
                                     std::vector<std::shared_ptr<op::Parameter>>{X});
    };

    for (auto i = 0; i < 100; i++)
    {
        auto x = rng.initialize(backend->make_primary_tensor_view<float>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<float>(manager, backend, make_graph, {x}, .01f, .01f));
    }
}

TEST(${BACKEND_NAME}, backwards_subtract)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape));
    auto x1 = rng.initialize(backend->make_primary_tensor_view<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        auto X1 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(X0 - X1, std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(${BACKEND_NAME}, backwards_sum_v2s)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape = Shape{8};
    auto x = rng.initialize(backend->make_primary_tensor_view<float>(shape));

    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Sum>(X, AxisSet{0}),
                                     std::vector<std::shared_ptr<op::Parameter>>{X});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(manager, backend, make_graph, {x}, .01f, .01f));
}

TEST(${BACKEND_NAME}, backwards_sum_m2s)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape = Shape{8, 9};
    auto x = rng.initialize(backend->make_primary_tensor_view<float>(shape));

    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Sum>(X, AxisSet{0, 1}),
                                     std::vector<std::shared_ptr<op::Parameter>>{X});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(manager, backend, make_graph, {x}, .01f, .01f));
}

TEST(${BACKEND_NAME}, backwards_sum_m2v_0)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape = Shape{8, 9};
    auto x = rng.initialize(backend->make_primary_tensor_view<float>(shape));

    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Sum>(X, AxisSet{0}),
                                     std::vector<std::shared_ptr<op::Parameter>>{X});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(manager, backend, make_graph, {x}, .01f, .01f));
}

TEST(${BACKEND_NAME}, backwards_sum_m2v_1)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape = Shape{8, 9};
    auto x = rng.initialize(backend->make_primary_tensor_view<float>(shape));

    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Sum>(X, AxisSet{1}),
                                     std::vector<std::shared_ptr<op::Parameter>>{X});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(manager, backend, make_graph, {x}, .01f, .01f));
}

TEST(${BACKEND_NAME}, backwards_tan)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    auto pi = 3.14159f;

    // Stay away from the asymptotes at 6 and 12 o'clock.
    auto slop = 0.1f;
    test::Uniform<float> rng_r(-pi / 2 + slop, pi / 2 - slop);
    test::Uniform<float> rng_l(pi / 2 + slop, (3 * pi) / 2 - slop);

    auto shape = Shape{2, 3};

    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Tan>(X),
                                     std::vector<std::shared_ptr<op::Parameter>>{X});
    };

    for (auto i = 0; i < 100; i++)
    {
        auto x_r = rng_r.initialize(backend->make_primary_tensor_view<float>(shape));

        EXPECT_TRUE(
            autodiff_numeric_compare<float>(manager, backend, make_graph, {x_r}, .01f, .01f));

        auto x_l = rng_l.initialize(backend->make_primary_tensor_view<float>(shape));

        EXPECT_TRUE(
            autodiff_numeric_compare<float>(manager, backend, make_graph, {x_l}, .01f, .01f));
    }
}

TEST(${BACKEND_NAME}, backwards_tanh)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-10.0f, 10.0f);
    auto shape = Shape{2, 3};
    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Tanh>(X),
                                     std::vector<std::shared_ptr<op::Parameter>>{X});
    };

    for (auto i = 0; i < 100; i++)
    {
        auto x = rng.initialize(backend->make_primary_tensor_view<float>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<float>(manager, backend, make_graph, {x}, .01f, .01f));
    }
}

TEST(${BACKEND_NAME}, backwards_abc)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape));
    auto x1 = rng.initialize(backend->make_primary_tensor_view<float>(shape));
    auto x2 = rng.initialize(backend->make_primary_tensor_view<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::f32, shape);
        auto X1 = make_shared<op::Parameter>(element::f32, shape);
        auto X2 = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>((X0 + X1) * X2,
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1, X2});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1, x2}, .01f, .01f));
}

TEST(${BACKEND_NAME}, backwards_reverse_3d_02)
{
    auto manager = runtime::Manager::get("${BACKEND_NAME}");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape = Shape{2, 4, 5};
    auto x = rng.initialize(backend->make_primary_tensor_view(element::f32, shape));

    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::f32, shape);
        return make_shared<Function>(make_shared<op::Reverse>(X, AxisSet{0, 2}),
                                     std::vector<std::shared_ptr<op::Parameter>>{X});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(manager, backend, make_graph, {x}, .01f, .01f));
}
