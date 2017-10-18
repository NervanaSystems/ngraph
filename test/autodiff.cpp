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
#include <memory>
#include <tuple>

#include "gtest/gtest.h"

#include "ngraph/autodiff/backprop_derivative.hpp"
#include "ngraph/autodiff/backprop_function.hpp"
#include "ngraph/autodiff/numeric_derivative.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/test/all_close.hpp"

using namespace std;
using namespace ngraph;

TEST(backwards, parameter)
{
    auto shape = Shape{2, 3};
    auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto Y = X0;
    auto C = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto DYDX0 = Y->backprop_node(X0, C);
    ASSERT_EQ(DYDX0, C);
}

TEST(backwards, add)
{
    auto shape = Shape{2, 3};
    auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto Y = X0 + X1;
    auto C = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto DYDX0 = Y->backprop_node(X0, C);
    auto DYDX1 = Y->backprop_node(X1, C);
    ASSERT_EQ(DYDX0, C);
    ASSERT_EQ(DYDX1, C);
}

TEST(backwards, multiply)
{
    auto shape = Shape{2, 3};
    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return make_shared<Function>(
            X0 * X1, nullptr, std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };

    auto manager = runtime::Manager::get("NGVM");
    auto backend = manager->allocate_backend();

    auto external = manager->compile(ngraph::autodiff::backprop_function(make_graph()));
    auto cf = backend->make_call_frame(external);

    auto x0 = backend->make_parameterized_tensor_view<element::Float32>(
        runtime::NDArray<float, 2>({{1, 3, 5}, {7, 9, 11}}));
    auto x1 = backend->make_parameterized_tensor_view<element::Float32>(
        runtime::NDArray<float, 2>({{0, 2, 4}, {6, 8, 10}}));
    auto c = backend->make_parameterized_tensor_view<element::Float32>(
        runtime::NDArray<float, 2>({{0, 0, 0}, {0, 0, 0}}));

    auto dx0 = backend->make_parameterized_tensor_view<element::Float32>(shape);
    auto dx1 = backend->make_parameterized_tensor_view<element::Float32>(shape);
    auto dx = backend->make_tuple({dx0, dx1});

    size_t n = x0->get_vector().size();
    vector<float> dx0_correct(n);
    vector<float> dx1_correct(n);
    for (size_t i = 0; i < n; i++)
    {
        c->get_vector().assign(n, 0);
        c->get_vector()[i] = 1;
        (*cf)({x0, x1, c}, {dx});
        dx0_correct.assign(n, 0);
        dx1_correct.assign(n, 0);
        dx0_correct[i] = x1->get_vector()[i];
        dx1_correct[i] = x0->get_vector()[i];
        ASSERT_EQ(dx0->get_vector(), dx0_correct);
        ASSERT_EQ(dx1->get_vector(), dx1_correct);
    }

    auto f_num = make_graph();
    auto results_num =
        autodiff::numeric_derivative<element::Float32>(manager, backend, f_num, {x0, x1}, .001f);
    auto f_sym = make_graph();
    auto results_sym =
        autodiff::backprop_derivative<element::Float32>(manager, backend, f_sym, {x0, x1});
    for (size_t i = 0; i < results_num.size(); ++i)
    {
        auto result_num = results_num[i];
        auto result_sym = results_sym[i];
        bool ac = test::all_close(result_num, result_sym, .01f, .01f);
        EXPECT_TRUE(ac);
    }
}
