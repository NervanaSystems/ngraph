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
#include "ngraph/test/all_close.hpp"
#include "ngraph/test/random.hpp"

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
    auto results_num =
        autodiff::numeric_derivative<element::Float32>(manager, backend, make_graph(), args, .001f);
    auto results_sym =
        autodiff::backprop_derivative<element::Float32>(manager, backend, make_graph(), args);
    return test::all_close(results_num, results_sym, .01f, .01f);
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

    auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto Y = X0;
    auto C = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto DYDX0 = Y->backprop_node(X0, C);
    ASSERT_EQ(DYDX0, C);
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

    auto results_num = autodiff::numeric_derivative<element::Float32>(
        manager, backend, make_graph(), {x0, x1}, .001f);
    auto results_sym =
        autodiff::backprop_derivative<element::Float32>(manager, backend, make_graph(), {x0, x1});
    EXPECT_TRUE(test::all_close(results_num, results_sym, .01f, .01f));
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
