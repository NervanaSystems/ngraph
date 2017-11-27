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
    auto results_num = autodiff::numeric_derivative<T>(manager, backend, make_graph(), args, .001f);
    auto results_sym = autodiff::backprop_derivative<T>(manager, backend, make_graph(), args);

    return test::all_close(results_num, results_sym, .01f, .01f);
}

TEST(backwards, add)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape));
    auto x1 = rng.initialize(backend->make_primary_tensor_view<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(
            X0 + X1, nullptr, std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(backwards, add_nested)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape));
    auto x1 = rng.initialize(backend->make_primary_tensor_view<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(
            (X0 + X1) + (X1 + X0), nullptr, std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(backwards, broadcast0)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape = Shape{3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(make_shared<op::Broadcast>(X0, Shape{2, 3}, AxisSet{0}),
                                     nullptr,
                                     std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(manager, backend, make_graph, {x0}, .01f, .01f));
}

TEST(backwards, broadcast1)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape = Shape{3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(make_shared<op::Broadcast>(X0, Shape{3, 2}, AxisSet{1}),
                                     nullptr,
                                     std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(manager, backend, make_graph, {x0}, .01f, .01f));
}

TEST(backwards, divide)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    test::Uniform<float> rng1(1.0f, 2.0f);
    test::Uniform<float> rng2(-2.0f, -1.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape));
    auto x1 = rng1.initialize(backend->make_primary_tensor_view<float>(shape));
    auto x2 = rng2.initialize(backend->make_primary_tensor_view<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(
            X0 / X1, nullptr, std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1}, .01f, .01f));
    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x2}, .01f, .01f));
}

TEST(backwards, dot_scalar_scalar)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape0 = Shape{};
    auto shape1 = Shape{};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape0));
    auto x1 = rng.initialize(backend->make_primary_tensor_view<float>(shape1));

    auto make_graph = [shape0, shape1]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape0);
        auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape1);
        return make_shared<Function>(make_shared<op::Dot>(X0, X1),
                                     nullptr,
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(backwards, dot_scalar_tensor)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape0 = Shape{};
    auto shape1 = Shape{3, 4};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape0));
    auto x1 = rng.initialize(backend->make_primary_tensor_view<float>(shape1));

    auto make_graph = [shape0, shape1]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape0);
        auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape1);
        return make_shared<Function>(make_shared<op::Dot>(X0, X1),
                                     nullptr,
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(backwards, dot_tensor_scalar)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape0 = Shape{3, 4};
    auto shape1 = Shape{};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape0));
    auto x1 = rng.initialize(backend->make_primary_tensor_view<float>(shape1));

    auto make_graph = [shape0, shape1]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape0);
        auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape1);
        return make_shared<Function>(make_shared<op::Dot>(X0, X1),
                                     nullptr,
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(backwards, dot_vector_vector)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape0 = Shape{3};
    auto shape1 = Shape{3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape0));
    auto x1 = rng.initialize(backend->make_primary_tensor_view<float>(shape1));

    auto make_graph = [shape0, shape1]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape0);
        auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape1);
        return make_shared<Function>(make_shared<op::Dot>(X0, X1),
                                     nullptr,
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(backwards, dot_tensor_vector)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape0 = Shape{4, 3};
    auto shape1 = Shape{3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape0));
    auto x1 = rng.initialize(backend->make_primary_tensor_view<float>(shape1));

    auto make_graph = [shape0, shape1]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape0);
        auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape1);
        return make_shared<Function>(make_shared<op::Dot>(X0, X1),
                                     nullptr,
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(backwards, dot_tensor2_tensor2)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape0 = Shape{4, 3};
    auto shape1 = Shape{3, 5};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape0));
    auto x1 = rng.initialize(backend->make_primary_tensor_view<float>(shape1));

    auto make_graph = [shape0, shape1]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape0);
        auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape1);
        return make_shared<Function>(make_shared<op::Dot>(X0, X1),
                                     nullptr,
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(backwards, exp)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(
            make_shared<op::Exp>(X0), nullptr, std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(manager, backend, make_graph, {x0}, .01f, .01f));
}

TEST(backwards, log)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(1.0f, 2.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(
            make_shared<op::Log>(X0), nullptr, std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(manager, backend, make_graph, {x0}, .01f, .01f));
}

TEST(backwards, maximum)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape));
    auto x1 = rng.initialize(backend->make_primary_tensor_view<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(make_shared<op::Maximum>(X0, X1),
                                     nullptr,
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(backwards, minimum)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape));
    auto x1 = rng.initialize(backend->make_primary_tensor_view<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(make_shared<op::Minimum>(X0, X1),
                                     nullptr,
                                     std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(backwards, multiply)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape));
    auto x1 = rng.initialize(backend->make_primary_tensor_view<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(
            X0 * X1, nullptr, std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(backwards, negative)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(-X0, nullptr, std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(manager, backend, make_graph, {x0}, .01f, .01f));
}

TEST(backwards, parameter)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape));
    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(X0, nullptr, std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(manager, backend, make_graph, {x0}, .01f, .01f));
}

TEST(backwards, power)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng_neg(-5.0f, -0.5f);
    test::Uniform<float> rng_pos(0.5f, 5.0f);
    auto shape = Shape{2, 3};

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(std::make_shared<op::Power>(X0, X1),
                                     nullptr,
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

TEST(backwards, reshape)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape = Shape{3, 4};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(make_shared<op::Reshape>(X0, AxisVector{1, 0}, Shape{4, 3}),
                                     nullptr,
                                     std::vector<std::shared_ptr<op::Parameter>>{X0});
    };
    EXPECT_TRUE(autodiff_numeric_compare<float>(manager, backend, make_graph, {x0}, .01f, .01f));
}

TEST(backwards, subtract)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape));
    auto x1 = rng.initialize(backend->make_primary_tensor_view<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(
            X0 - X1, nullptr, std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1}, .01f, .01f));
}

TEST(backwards, abc)
{
    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    test::Uniform<float> rng(-1.0f, 1.0f);
    auto shape = Shape{2, 3};
    auto x0 = rng.initialize(backend->make_primary_tensor_view<float>(shape));
    auto x1 = rng.initialize(backend->make_primary_tensor_view<float>(shape));
    auto x2 = rng.initialize(backend->make_primary_tensor_view<float>(shape));

    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        auto X2 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(
            (X0 + X1) * X2, nullptr, std::vector<std::shared_ptr<op::Parameter>>{X0, X1, X2});
    };
    EXPECT_TRUE(
        autodiff_numeric_compare<float>(manager, backend, make_graph, {x0, x1, x2}, .01f, .01f));
}
