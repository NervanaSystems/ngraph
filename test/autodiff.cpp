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

#include "ngraph/autodiff/backprop_derivative.hpp"
#include "ngraph/autodiff/backprop_function.hpp"
#include "ngraph/autodiff/numeric_derivative.hpp"
#include "ngraph/ngraph.hpp"
#include "util/all_close.hpp"
#include "util/random.hpp"

using namespace std;
using namespace ngraph;

template <typename ET>
bool autodiff_numeric_compare(
    const std::shared_ptr<runtime::Manager>& manager,
    const std::shared_ptr<runtime::Backend>& backend,
    std::function<std::shared_ptr<Function>()> make_graph,
    const std::vector<std::shared_ptr<runtime::ParameterizedTensorView<ET>>>& args,
    typename ET::type rtol,
    typename ET::type atol)
{
    std::vector<std::shared_ptr<runtime::TensorView>> args_as_tv;

    for (auto arg : args)
    {
        args_as_tv.push_back(static_pointer_cast<runtime::TensorView>(arg));
    }

    auto f = make_graph();
    auto results_num =
        autodiff::numeric_derivative<ET>(manager, backend, f, args_as_tv, .001f, f->get_parameters());

    auto g = make_graph();
    auto results_sym =
        autodiff::backprop_derivative<ET>(manager, backend, g, args_as_tv, g->get_parameters());

    return test::all_close(results_num, results_sym, rtol, atol);
}

template <typename ET>
bool autodiff_numeric_compare_selective(
    const std::shared_ptr<runtime::Manager>& manager,
    const std::shared_ptr<runtime::Backend>& backend,
    std::function<std::shared_ptr<Function>()> make_graph,
    const std::vector<std::shared_ptr<runtime::TensorView>>& args,
    typename ET::type rtol,
    typename ET::type atol,
    const std::vector<bool>& indep_param_mask)
{
    std::vector<std::shared_ptr<runtime::TensorView>> args_as_tv;

    for (auto arg : args)
    {
        args_as_tv.push_back(static_pointer_cast<runtime::TensorView>(arg));
    }

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
        autodiff::numeric_derivative<ET>(manager, backend, f, args_as_tv, .001f, f_indep_params);

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

    auto results_sym =
        autodiff::backprop_derivative<ET>(manager, backend, g, args_as_tv, g_indep_params);

    return test::all_close(results_num, results_sym, rtol, atol);
}

TEST(backwards, abs)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    // The numeric derivative and the symbolic one may disagree around 0, so we will dance around
    // that point by skipping (-0.01,0.01).
    test::Uniform<element::Float32> rng_neg(-1.0f, 0.01f);
    test::Uniform<element::Float32> rng_pos(0.01f, 1.0f);
    auto shape = Shape{2, 3};

    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(
            make_shared<op::Abs>(X), nullptr, std::vector<std::shared_ptr<op::Parameter>>{X});
    };

    for (auto i = 0; i < 100; i++)
    {
        auto x_neg =
            rng_neg.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<element::Float32>(
            manager, backend, make_graph, {x_neg}, .01f, .01f));

        auto x_pos =
            rng_pos.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<element::Float32>(
            manager, backend, make_graph, {x_pos}, .01f, .01f));
    }
}

TEST(backwards, add)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<element::Float32> rng(-1.0f, 1.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));
    auto x1 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(
            X0 + X1, nullptr, std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(autodiff_numeric_compare<element::Float32>(
        manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(backwards, add_nested)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<element::Float32> rng(-1.0f, 1.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));
    auto x1 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(
            (X0 + X1) + (X1 + X0), nullptr, std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(autodiff_numeric_compare<element::Float32>(
        manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(backwards, broadcast0)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<element::Float32> rng(-1.0f, 1.0f);
    auto shape = Shape{3};
    auto x0 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(make_shared<op::Broadcast>(X0, Shape{2, 3}, AxisSet{0}),
                                     nullptr,
                                     std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<element::Float32>(manager, backend, make_graph, {x0}, .01f, .01f));
}

TEST(backwards, broadcast1)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<element::Float32> rng(-1.0f, 1.0f);
    auto shape = Shape{3};
    auto x0 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(make_shared<op::Broadcast>(X0, Shape{3, 2}, AxisSet{1}),
                                     nullptr,
                                     std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<element::Float32>(manager, backend, make_graph, {x0}, .01f, .01f));
}

TEST(backwards, ceiling)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    // The numeric derivative and the symbolic one may disagree near integers, so we will dance around
    // them.
    test::Uniform<element::Float32> rng_minusone(-0.95f, -0.05f);
    test::Uniform<element::Float32> rng_plusone(0.05f, 0.95f);
    test::Uniform<element::Float32> rng_plustwo(1.05f, 1.95f);
    auto shape = Shape{2, 3};

    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(
            make_shared<op::Ceiling>(X), nullptr, std::vector<std::shared_ptr<op::Parameter>>{X});
    };

    for (auto i = 0; i < 100; i++)
    {
        auto x_minusone =
            rng_minusone.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<element::Float32>(
            manager, backend, make_graph, {x_minusone}, .01f, .01f));

        auto x_plusone =
            rng_plusone.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<element::Float32>(
            manager, backend, make_graph, {x_plusone}, .01f, .01f));

        auto x_plustwo =
            rng_plustwo.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<element::Float32>(
            manager, backend, make_graph, {x_plustwo}, .01f, .01f));
    }
}

TEST(backwards, cos)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<element::Float32> rng(-10.0f, 10.0f);
    auto shape = Shape{2, 3};
    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(
            make_shared<op::Cos>(X), nullptr, std::vector<std::shared_ptr<op::Parameter>>{X});
    };

    for (auto i = 0; i < 100; i++)
    {
        auto x = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<element::Float32>(
            manager, backend, make_graph, {x}, .01f, .01f));
    }
}

TEST(backwards, divide)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<element::Float32> rng(-1.0f, 1.0f);
    test::Uniform<element::Float32> rng1(1.0f, 2.0f);
    test::Uniform<element::Float32> rng2(-2.0f, -1.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));
    auto x1 = rng1.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));
    auto x2 = rng2.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(
            X0 / X1, nullptr, std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(autodiff_numeric_compare<element::Float32>(
        manager, backend, make_graph, {x0, x1}, .01f, .01f));
    EXPECT_TRUE(autodiff_numeric_compare<element::Float32>(
        manager, backend, make_graph, {x0, x2}, .01f, .01f));
}

TEST(backwards, dot_scalar_scalar)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<element::Float32> rng(-1.0f, 1.0f);
    auto shape0 = Shape{};
    auto shape1 = Shape{};
    auto x0 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape0));
    auto x1 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape1));

    auto make_graph = [shape0, shape1]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape0);
        auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape1);
        return make_shared<Function>(make_shared<op::Dot>(X0, X1),
                                     nullptr,
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(autodiff_numeric_compare<element::Float32>(
        manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(backwards, dot_scalar_tensor)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<element::Float32> rng(-1.0f, 1.0f);
    auto shape0 = Shape{};
    auto shape1 = Shape{3, 4};
    auto x0 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape0));
    auto x1 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape1));

    auto make_graph = [shape0, shape1]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape0);
        auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape1);
        return make_shared<Function>(make_shared<op::Dot>(X0, X1),
                                     nullptr,
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(autodiff_numeric_compare<element::Float32>(
        manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(backwards, dot_tensor_scalar)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<element::Float32> rng(-1.0f, 1.0f);
    auto shape0 = Shape{3, 4};
    auto shape1 = Shape{};
    auto x0 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape0));
    auto x1 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape1));

    auto make_graph = [shape0, shape1]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape0);
        auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape1);
        return make_shared<Function>(make_shared<op::Dot>(X0, X1),
                                     nullptr,
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(autodiff_numeric_compare<element::Float32>(
        manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(backwards, dot_vector_vector)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<element::Float32> rng(-1.0f, 1.0f);
    auto shape0 = Shape{3};
    auto shape1 = Shape{3};
    auto x0 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape0));
    auto x1 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape1));

    auto make_graph = [shape0, shape1]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape0);
        auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape1);
        return make_shared<Function>(make_shared<op::Dot>(X0, X1),
                                     nullptr,
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(autodiff_numeric_compare<element::Float32>(
        manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(backwards, dot_tensor_vector)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<element::Float32> rng(-1.0f, 1.0f);
    auto shape0 = Shape{4, 3};
    auto shape1 = Shape{3};
    auto x0 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape0));
    auto x1 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape1));

    auto make_graph = [shape0, shape1]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape0);
        auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape1);
        return make_shared<Function>(make_shared<op::Dot>(X0, X1),
                                     nullptr,
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(autodiff_numeric_compare<element::Float32>(
        manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(backwards, dot_tensor2_tensor2)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<element::Float32> rng(-1.0f, 1.0f);
    auto shape0 = Shape{4, 3};
    auto shape1 = Shape{3, 5};
    auto x0 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape0));
    auto x1 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape1));

    auto make_graph = [shape0, shape1]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape0);
        auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape1);
        return make_shared<Function>(make_shared<op::Dot>(X0, X1),
                                     nullptr,
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(autodiff_numeric_compare<element::Float32>(
        manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(backwards, exp)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<element::Float32> rng(-1.0f, 1.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(
            make_shared<op::Exp>(X0), nullptr, std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<element::Float32>(manager, backend, make_graph, {x0}, .01f, .01f));
}

TEST(backwards, floor)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    // The numeric derivative and the symbolic one may disagree near integers, so we will dance around
    // them.
    test::Uniform<element::Float32> rng_minusone(-0.95f, -0.05f);
    test::Uniform<element::Float32> rng_plusone(0.05f, 0.95f);
    test::Uniform<element::Float32> rng_plustwo(1.05f, 1.95f);
    auto shape = Shape{2, 3};

    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(
            make_shared<op::Floor>(X), nullptr, std::vector<std::shared_ptr<op::Parameter>>{X});
    };

    for (auto i = 0; i < 100; i++)
    {
        auto x_minusone =
            rng_minusone.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<element::Float32>(
            manager, backend, make_graph, {x_minusone}, .01f, .01f));

        auto x_plusone =
            rng_plusone.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<element::Float32>(
            manager, backend, make_graph, {x_plusone}, .01f, .01f));

        auto x_plustwo =
            rng_plustwo.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<element::Float32>(
            manager, backend, make_graph, {x_plustwo}, .01f, .01f));
    }
}

TEST(backwards, log)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<element::Float32> rng(1.0f, 2.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(
            make_shared<op::Log>(X0), nullptr, std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<element::Float32>(manager, backend, make_graph, {x0}, .01f, .01f));
}

TEST(backwards, maximum)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<element::Float32> rng(-1.0f, 1.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));
    auto x1 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(make_shared<op::Maximum>(X0, X1),
                                     nullptr,
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(autodiff_numeric_compare<element::Float32>(
        manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(backwards, minimum)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<element::Float32> rng(-1.0f, 1.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));
    auto x1 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(make_shared<op::Minimum>(X0, X1),
                                     nullptr,
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(autodiff_numeric_compare<element::Float32>(
        manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(backwards, multiply)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<element::Float32> rng(-1.0f, 1.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));
    auto x1 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(
            X0 * X1, nullptr, std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(autodiff_numeric_compare<element::Float32>(
        manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(backwards, negative)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<element::Float32> rng(-1.0f, 1.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(-X0, nullptr, std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<element::Float32>(manager, backend, make_graph, {x0}, .01f, .01f));
}

TEST(backwards, parameter)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<element::Float32> rng(-1.0f, 1.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));
    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(X0, nullptr, std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<element::Float32>(manager, backend, make_graph, {x0}, .01f, .01f));
}

TEST(backwards, reshape)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<element::Float32> rng(-1.0f, 1.0f);
    auto shape = Shape{3, 4};
    auto x0 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(make_shared<op::Reshape>(X0, AxisVector{1, 0}, Shape{4, 3}),
                                     nullptr,
                                     std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<element::Float32>(manager, backend, make_graph, {x0}, .01f, .01f));
}

TEST(backwards, select)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<element::Float32> rng(-10.0f, 10.0f);
    auto shape = Shape{2, 3};
    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Bool::element_type(), shape);
        auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        auto X2 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(
            make_shared<op::Select>(X0,X1,X2), nullptr, std::vector<std::shared_ptr<op::Parameter>>{X0,X1,X2});
    };

    for (auto i = 0; i < 100; i++)
    {
        auto x0 = backend->make_parameterized_tensor_view<element::Bool>(shape);
        *x0 = vector<char>{0, 1, 0, 1, 0, 1};
        auto x1 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));
        auto x2 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));

        EXPECT_TRUE(autodiff_numeric_compare_selective<element::Float32>(
            manager, backend, make_graph, {x0, x1, x2}, .01f, .01f, std::vector<bool>{false,true,true}));
    }
}

TEST(backwards, select_nested)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<element::Float32> rng(-10.0f, 10.0f);
    auto shape = Shape{2, 3};
    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Bool::element_type(), shape);
        auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        auto X2 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(
            make_shared<op::Select>(X0,X1+X2,X2-X1), nullptr, std::vector<std::shared_ptr<op::Parameter>>{X0,X1,X2});
    };

    for (auto i = 0; i < 100; i++)
    {
        auto x0 = backend->make_parameterized_tensor_view<element::Bool>(shape);
        *x0 = vector<char>{0, 1, 0, 1, 0, 1};
        auto x1 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));
        auto x2 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));

        EXPECT_TRUE(autodiff_numeric_compare_selective<element::Float32>(
            manager, backend, make_graph, {x0, x1, x2}, .01f, .01f, std::vector<bool>{false,true,true}));
    }
}

TEST(backwards, sign)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    // The numeric derivative and the symbolic one may disagree around 0, so we will dance around
    // that point by skipping (-0.01,0.01).
    test::Uniform<element::Float32> rng_neg(-1.0f, 0.01f);
    test::Uniform<element::Float32> rng_pos(0.01f, 1.0f);
    auto shape = Shape{2, 3};

    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(
            make_shared<op::Sign>(X), nullptr, std::vector<std::shared_ptr<op::Parameter>>{X});
    };

    for (auto i = 0; i < 100; i++)
    {
        auto x_neg =
            rng_neg.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<element::Float32>(
            manager, backend, make_graph, {x_neg}, .01f, .01f));

        auto x_pos =
            rng_pos.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<element::Float32>(
            manager, backend, make_graph, {x_pos}, .01f, .01f));
    }
}

TEST(backwards, sin)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<element::Float32> rng(-10.0f, 10.0f);
    auto shape = Shape{2, 3};
    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(
            make_shared<op::Sin>(X), nullptr, std::vector<std::shared_ptr<op::Parameter>>{X});
    };

    for (auto i = 0; i < 100; i++)
    {
        auto x = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<element::Float32>(
            manager, backend, make_graph, {x}, .01f, .01f));
    }
}

TEST(backwards, subtract)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<element::Float32> rng(-1.0f, 1.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));
    auto x1 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(
            X0 - X1, nullptr, std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(autodiff_numeric_compare<element::Float32>(
        manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(backwards, tan)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    auto pi = 3.14159f;

    // Stay away from the asymptotes at 6 and 12 o'clock.
    auto slop = 0.1f;
    test::Uniform<element::Float32> rng_r(-pi / 2 + slop, pi / 2 - slop);
    test::Uniform<element::Float32> rng_l(pi / 2 + slop, (3 * pi) / 2 - slop);

    auto shape = Shape{2, 3};

    auto make_graph = [shape]() {
        auto X = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(
            make_shared<op::Tan>(X), nullptr, std::vector<std::shared_ptr<op::Parameter>>{X});
    };

    for (auto i = 0; i < 100; i++)
    {
        auto x_r =
            rng_r.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<element::Float32>(
            manager, backend, make_graph, {x_r}, .01f, .01f));

        auto x_l =
            rng_l.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));

        EXPECT_TRUE(autodiff_numeric_compare<element::Float32>(
            manager, backend, make_graph, {x_l}, .01f, .01f));
    }
}

TEST(backwards, abc)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<element::Float32> rng(-1.0f, 1.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));
    auto x1 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));
    auto x2 = rng.initialize(backend->make_parameterized_tensor_view<element::Float32>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        auto X2 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(
            (X0 + X1) * X2, nullptr, std::vector<std::shared_ptr<op::Parameter>>{X0, X1, X2});
    };
    EXPECT_TRUE(autodiff_numeric_compare<element::Float32>(
        manager, backend, make_graph, {x0, x1, x2}, .01f, .01f));
}
