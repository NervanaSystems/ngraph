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
#include <cstdio>
#include <iostream>
#include <list>
#include <memory>

#include "gtest/gtest.h"
#include "ngraph/file_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/pass/algebraic_simplification.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/any.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/serializer.hpp"
#include "util/matcher.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

TEST(pattern, algebraic_simplification_add_types_shapes)
{
    Shape shapes[] = {Shape{}, Shape{2, 2}, Shape{3, 3, 3}};
    element::Type types[] = {element::i32, element::f32, element::f64};
    for (auto type : types)
    {
        for (auto shape : shapes)
        {
            pass::Manager pass_manager;
            pass_manager.register_pass<pass::VisualizeTree>("before.pdf");
            pass_manager.register_pass<pass::AlgebraicSimplification>();
            pass_manager.register_pass<pass::VisualizeTree>("after.pdf");

            auto a = make_shared<op::Parameter>(type, shape);
            auto b = make_shared<op::Parameter>(type, shape);
            auto c = make_shared<op::Parameter>(type, shape);
            auto iconst0 = ngraph::make_constant_from_string("0", type, shape);
            auto add_a_0 = a + iconst0;
            auto add_a_0_0 = add_a_0 + iconst0;
            auto add_b_0 = b + iconst0;
            auto add_b_0_0 = add_b_0 + iconst0;

            auto f = std::make_shared<Function>(ngraph::NodeVector{a, b, add_a_0_0, c, add_b_0_0},
                                                op::ParameterVector{a, b, c});
            pass_manager.run_passes(f);

            ASSERT_EQ(count_ops_of_type<op::Add>(f), 0);
            auto expected = ngraph::NodeVector{a, b, a, c, b};
            auto results = f->get_results();
            for (size_t i = 0; i < results.size(); i++)
            {
                ASSERT_EQ(expected.at(i), results.at(i)->get_argument(0));
            }
        }
    }
}

TEST(pattern, algebraic_simplification_add_broadcast)
{
    Shape shape{2, 2};
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::VisualizeTree>("before.pdf");
    pass_manager.register_pass<pass::AlgebraicSimplification>();
    pass_manager.register_pass<pass::VisualizeTree>("after.pdf");

    auto a = make_shared<op::Parameter>(element::i32, shape);
    auto b = make_shared<op::Parameter>(element::i32, shape);
    auto c = make_shared<op::Parameter>(element::i32, shape);
    auto iconst0 = ngraph::make_zero(element::i32, Shape{});
    auto const_broadcast = make_shared<op::Broadcast>(iconst0, shape, AxisSet{0, 1});
    auto add_a_0 = a + const_broadcast;
    auto add_a_0_0 = add_a_0 + const_broadcast;
    auto add_b_0 = b + const_broadcast;
    auto add_b_0_0 = add_b_0 + const_broadcast;

    auto f = std::make_shared<Function>(ngraph::NodeVector{a, b, add_a_0_0, c, add_b_0_0},
                                        op::ParameterVector{a, b, c});
    pass_manager.run_passes(f);

    ASSERT_EQ(count_ops_of_type<op::Add>(f), 0);
    auto expected = ngraph::NodeVector{a, b, a, c, b};
    auto results = f->get_results();
    for (size_t i = 0; i < results.size(); i++)
    {
        ASSERT_EQ(expected.at(i), results.at(i)->get_argument(0));
    }
}

TEST(pattern, algebraic_simplification_multiply_zero_one)
{
    Shape shapes[] = {Shape{}, Shape{2, 2}, Shape{3, 3, 3}};
    element::Type types[] = {element::i32, element::f32, element::f64};
    const size_t NUM_TESTS = 2;
    auto type = element::i32;
    Shape shape{};

    NodeVector consts = {ngraph::make_constant_from_string("0", type, shape),
                         ngraph::make_constant_from_string("1", type, shape)};

    auto a = make_shared<op::Parameter>(type, shape);
    auto b = make_shared<op::Parameter>(type, shape);
    auto c = make_shared<op::Parameter>(type, shape);
    std::string vals[] = {"0", "1"};
    NodeVector expected_results[] = {ngraph::NodeVector{a, b, consts.at(0), c, consts.at(0)},
                                     ngraph::NodeVector{a, b, a, c, b}};

    for (size_t j = 0; j < NUM_TESTS; j++)
    {
        pass::Manager pass_manager;
        pass_manager.register_pass<pass::VisualizeTree>("before.pdf");
        pass_manager.register_pass<pass::AlgebraicSimplification>();
        pass_manager.register_pass<pass::VisualizeTree>("after.pdf");

        auto iconst = consts.at(j);
        auto multiply_a_0 = a * iconst;
        auto multiply_a_0_0 = multiply_a_0 * iconst;
        auto multiply_b_0 = b * iconst;
        auto multiply_b_0_0 = multiply_b_0 * iconst;

        auto f =
            std::make_shared<Function>(ngraph::NodeVector{a, b, multiply_a_0_0, c, multiply_b_0_0},
                                       op::ParameterVector{a, b, c});
        pass_manager.run_passes(f);

        ASSERT_EQ(count_ops_of_type<op::Multiply>(f), 0);
        auto expected = expected_results[j];
        auto results = f->get_results();
        for (size_t i = 0; i < results.size(); i++)
        {
            ASSERT_EQ(expected.at(i), results.at(i)->get_argument(0));
        }
    }
}
