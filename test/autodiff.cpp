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

#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;

TEST(backwards, parameter)
{
    auto shape = Shape{2, 3};
    auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto Y = X0;
    auto C = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto DYDX0 = Y->backwards_derivative(X0, C);
    ASSERT_EQ(DYDX0, C);
}

TEST(backwards, add)
{
    auto shape = Shape{2, 3};
    auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto Y = X0 + X1;
    auto C = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto DYDX0 = Y->backwards_derivative(X0, C);
    auto DYDX1 = Y->backwards_derivative(X1, C);
    ASSERT_EQ(DYDX0, C);
    ASSERT_EQ(DYDX1, C);
}

// Returns (dy/(dXs))(C, Xs)
shared_ptr<Function> derivative(const std::shared_ptr<Node>& Y,
                                const std::vector<std::shared_ptr<op::Parameter>> Xs)
{
    auto Y_tv_type = dynamic_pointer_cast<const TensorViewType>(Y->get_value_type());
    auto C = make_shared<op::Parameter>(Y_tv_type->get_element_type(), Y_tv_type->get_shape());
    std::vector<std::shared_ptr<Node>> dYdXs(Xs.size());
    transform(Xs.begin(), Xs.end(), dYdXs.begin(), [C, Y](const std::shared_ptr<Node>& X) {
        return Y->backwards_derivative(X, C);
    });
    auto result = make_shared<op::Tuple>(dYdXs);
    std::vector<std::shared_ptr<op::Parameter>> args;
    args.push_back(C);
    args.insert(args.end(), Xs.begin(), Xs.end());
    return make_shared<Function>(result, result->get_value_type(), args);
}

TEST(backwards, multiply)
{
    auto shape = Shape{2, 3};
    auto make_graph = [shape]() {
        auto X0 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        auto X1 = make_shared<op::Parameter>(element::Float32::element_type(), shape);
        return std::make_tuple(X0 * X1, std::vector<std::shared_ptr<op::Parameter>>{X0, X1});
    };
    auto val_and_vars = make_graph();
    auto Y = std::get<0>(val_and_vars);
    auto vars = std::get<1>(val_and_vars);
    auto f = derivative(Y, vars);

    auto manager = runtime::Manager::get("NGVM");
    auto external = manager->compile(f);
    auto backend = manager->allocate_backend();
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
        (*cf)({c, x0, x1}, {dx});
        dx0_correct.assign(n, 0);
        dx1_correct.assign(n, 0);
        dx0_correct[i] = x1->get_vector()[i];
        dx1_correct[i] = x0->get_vector()[i];
        ASSERT_EQ(dx0->get_vector(), dx0_correct);
        ASSERT_EQ(dx1->get_vector(), dx1_correct);
    }
}
