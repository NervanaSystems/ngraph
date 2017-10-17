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

#include <memory>
#include <list>
#include <cstdio>
#include <iostream>
#include "gtest/gtest.h"
#include "ngraph/log.hpp"

#include "ngraph/ngraph.hpp"

#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/any.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pass/graph_rewrite.hpp"

using namespace ngraph;
using namespace std;

/*
TEST(graph_rewrite, basic)
{
    auto shape = Shape{1};
    auto a = make_shared<op::Parameter>(element::Int32::element_type(), shape);
    auto b = make_shared<op::Parameter>(element::Int32::element_type(), shape);
    auto c = make_shared<op::Parameter>(element::Int32::element_type(), shape);
    auto d = make_shared<op::Parameter>(element::Int32::element_type(), shape);
    
    auto pattern = std::make_shared<pattern::op::Label>();
    auto sum = b + a;
    auto pattern_mul = pattern_sum * a;
    auto mul = sum * a;

    ngraph::pattern::gr_callback_fn callback = [pattern](pattern::Matcher& m)
    {
        pattern->get_binded_node()
        ngraph::pass::GraphRewrite::replace_node(m.match_root(), m.pattern_node());
    };
    a * 1 = a;
    auto m = make_shared<pattern::Matcher>(pattern + a);
    auto pattern = std::make_shared<pattern::op::Pattern>(); //marker
    ASSERT_TRUE(m->match(pattern_mul, mul));

    ngraph::pass::GraphRewrite gr;

    std::list<std::shared_ptr<Node>> nodes{ b, a, sum, mul };
    gr.run_on_call_graph(nodes);
    ASSERT_EQ(mul->get_arguments().at(0), pattern);
    auto& sum_users = sum->users();
    ASSERT_TRUE(sum_users.find(mul.get()) == sum_users.end());
}
*/
TEST(pattern, op_op)
{
    auto shape = Shape{1};

    auto a = make_shared<op::Parameter>(element::Int32::element_type(), shape);
    pattern::Matcher n(nullptr);
    ASSERT_TRUE(n.match(a, a));

    auto abs = make_shared<op::Abs>(a);
    auto any = std::make_shared<pattern::op::Any>(a); 
    ASSERT_TRUE(n.match(any, abs));

    auto any_false = std::make_shared<pattern::op::Any>(a, [](std::shared_ptr<Node> n) {return false;});
    ASSERT_TRUE(n.match(any_false, a));

    auto pattern = std::make_shared<pattern::op::Label>();
    ASSERT_TRUE(n.match(pattern, a));
    ASSERT_EQ(pattern->get_binded_node(), a);
    
    auto pattern_false = std::make_shared<pattern::op::Label>([](std::shared_ptr<Node> n) {return false;});
    ASSERT_FALSE(n.match(pattern_false, a));
    
    auto b = make_shared<op::Parameter>(element::Int32::element_type(), shape);
    ASSERT_FALSE(n.match(a, b));
    ASSERT_FALSE(n.match(abs + b, b + b));
    ASSERT_TRUE(n.match(any + b, abs + b));

    ASSERT_TRUE(n.match(pattern + b, abs + b));
    ASSERT_EQ(pattern->get_binded_node(), abs);

    ASSERT_TRUE(n.match(b + pattern, abs + b));
    ASSERT_EQ(pattern->get_binded_node(), abs);

    auto c = make_shared<op::Parameter>(element::Int32::element_type(), shape);
    ASSERT_TRUE(n.match(c * (b + pattern), c * (abs + b)));
    ASSERT_EQ(pattern->get_binded_node(), abs);

    ASSERT_TRUE(n.match(c * (any + b), c * (abs + b))); //nested any
    ASSERT_TRUE(n.match(c * (any + b), (b + abs) * c)); //permutations w/ any
    ASSERT_TRUE(n.match(c * (any_false + b), c * (a + b))); //nested any
    ASSERT_TRUE(n.match(c * (any_false + b), (b + a) * c)); //permutations w/ any_false 
}
