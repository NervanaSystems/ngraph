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

#include <memory>

#include "gtest/gtest.h"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/inliner.hpp"
#include "ngraph/pass/manager.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(inline, basic)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto C = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>((A + B) * C, op::ParameterVector{A, B, C});

    auto X = make_shared<op::Parameter>(element::f32, shape);
    auto Y = make_shared<op::Parameter>(element::f32, shape);
    auto Z = make_shared<op::Parameter>(element::f32, shape);
    auto fc1 = make_shared<op::FunctionCall>(f, NodeVector{X, Y, Z});
    auto fc2 = make_shared<op::FunctionCall>(f, NodeVector{X, Y, Z});
    auto g = make_shared<Function>(fc1 + fc2, op::ParameterVector{X, Y, Z});

    auto ih = std::make_shared<ngraph::pass::InlineSmallCalls>(10, 1);
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::Inliner>(ih);
    auto bc = g->get_ops().size();
    pass_manager.run_passes(g);
    auto ac = g->get_ops().size();
    ASSERT_EQ(count_ops_of_type<op::FunctionCall>(g), 0); // check that FunctionCalls disappear
    ASSERT_LT(bc, ac);                                    // we should get more ops after inlining
}

TEST(inline, recursive)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>((A + B), op::ParameterVector{A, B});

    auto X = make_shared<op::Parameter>(element::f32, shape);
    auto Y = make_shared<op::Parameter>(element::f32, shape);

    auto fc1 = make_shared<op::FunctionCall>(f, NodeVector{X, Y});
    auto g = make_shared<Function>(make_shared<op::Negative>(fc1), op::ParameterVector{X, Y});

    auto P1 = make_shared<op::Parameter>(element::f32, shape);
    auto P2 = make_shared<op::Parameter>(element::f32, shape);
    auto P3 = make_shared<op::Parameter>(element::f32, shape);
    auto fc2 = make_shared<op::FunctionCall>(g, NodeVector{P1, P2});

    auto e = make_shared<Function>(fc2 * P3, op::ParameterVector{P1, P2, P3});
    auto ih = std::make_shared<ngraph::pass::InlineSmallCalls>(15, 2);
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::Inliner>(ih);
    auto bce = e->get_ops().size();
    pass_manager.run_passes(e);
    auto ace = e->get_ops().size();
    ASSERT_EQ(count_ops_of_type<op::FunctionCall>(g), 0); // check that FunctionCalls disappear
    ASSERT_EQ(count_ops_of_type<op::Add>(g), 1);          // FunctionCall is replaced w/ Add
    ASSERT_EQ(count_ops_of_type<op::FunctionCall>(e), 0);
    ASSERT_LT(bce, ace); // we should get more ops after inlining
}
