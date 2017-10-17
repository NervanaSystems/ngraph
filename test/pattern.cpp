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

class TestMatcher : public ngraph::pattern::Matcher
{
public:
    bool compare_nodes(const std::shared_ptr<ngraph::Node>& pattern_node,
                       const std::shared_ptr<ngraph::Node>& graph_node);

    virtual void on_match_class(const std::shared_ptr<ngraph::Node>& pattern_node,
                                const std::shared_ptr<ngraph::Node>& graph_node,
                                bool is_match);

    std::shared_ptr<ngraph::Node> get_pattern_node() const { return m_pattern_node; }
    std::shared_ptr<ngraph::Node> get_graph_node() const { return m_graph_node; }
    bool get_is_valid() const { return m_is_valid; }
protected:
    bool m_is_valid;
    bool m_is_match;
    std::shared_ptr<ngraph::Node> m_pattern_node;
    std::shared_ptr<ngraph::Node> m_graph_node;
};

void TestMatcher::on_match_class(const std::shared_ptr<ngraph::Node>& pattern_node,
                                 const std::shared_ptr<ngraph::Node>& graph_node,
                                 bool is_match)
{
    m_pattern_node = pattern_node;
    m_graph_node = graph_node;
    m_is_match = is_match;
    m_is_valid = true;
}

bool TestMatcher::compare_nodes(const std::shared_ptr<ngraph::Node>& pattern_node,
                                const std::shared_ptr<ngraph::Node>& graph_node)
{
    m_pattern_node = nullptr;
    m_graph_node = nullptr;
    m_is_match = false;
    m_is_valid = false;
    pattern_node->match_class(*this, graph_node);
    return m_is_match;
}

using namespace ngraph;
using namespace std;

/*
TEST(graph_rewrite, basic)
{
    auto shape = Shape{ 1 };
    auto a = make_shared<op::Parameter>(element::Int32::element_type(), shape);
    auto b = make_shared<op::Parameter>(element::Int32::element_type(), shape);
    auto any = std::make_shared<pattern::op::Any>(); //Abs
    auto sum = b + a;
    auto pattern_sum = any + a;
    auto pattern_mul = pattern_sum * a;
    auto mul = sum * a;
    auto m = make_shared<pattern::Matcher>(pattern_sum);
    auto pattern = std::make_shared<pattern::op::Pattern>(); //marker
    ASSERT_TRUE(m->match(pattern_mul, mul));
    //m->reset();
    ngraph::pass::GraphRewrite gr;
    ngraph::pattern::gr_callback_fn callback = [pattern](pattern::Matcher& m)
    {
        ngraph::pass::GraphRewrite::replace_node(m.match_root(), m.pattern_node());
    };
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
    //n.reset();

    auto pattern = std::make_shared<pattern::op::Label>();
    ASSERT_TRUE(n.match(pattern, a));
    ASSERT_EQ(pattern->get_binded_node(), a);
    
    auto pattern_false = std::make_shared<pattern::op::Label>([](std::shared_ptr<Node> n) {return false;});
    //n.reset();
    ASSERT_FALSE(n.match(pattern_false, a));
    
    auto b = make_shared<op::Parameter>(element::Int32::element_type(), shape);
    //n.reset();
    ASSERT_FALSE(n.match(a, b));

    auto sum = abs + b;
    auto sum_2b = b + b;
    //n.reset();
    ASSERT_FALSE(n.match(sum, sum_2b));
   
    auto any_sum = any + b;
    //n.reset();
    ASSERT_TRUE(n.match(any_sum, sum));

    //n.reset();
    auto pattern_sum = pattern + b;
    ASSERT_TRUE(n.match(pattern_sum, sum));
    ASSERT_EQ(pattern->get_binded_node(), abs);

    
    //n.reset();
    //pattern->reset();
    auto pattern_sum_perm = b + pattern;
    ASSERT_TRUE(n.match(pattern_sum_perm, sum));
    ASSERT_EQ(pattern->get_binded_node(), abs);

    
    auto c = make_shared<op::Parameter>(element::Int32::element_type(), shape);
    //pattern->reset();
    auto pattern_mul_perm = c * pattern_sum_perm;
    auto mul = c * sum;
    ASSERT_TRUE(n.match(pattern_mul_perm, mul));
    ASSERT_EQ(pattern->get_binded_node(), abs);


    ASSERT_TRUE(n.match(c * (any + b), c * (abs + b))); //nested any
    ASSERT_TRUE(n.match(c * (any + b), (b + abs) * c)); //permutations w/ any
    ASSERT_TRUE(n.match(c * (any_false + b), c * (a + b))); //nested any
    ASSERT_TRUE(n.match(c * (any_false + b), (b + a) * c)); //permutations w/ any_false 
    /*
    //pattern->reset();
    
    auto mula = a*b;
    auto mulc = c*b;
    auto sumac = mula + mulc;

    auto any_pattern_mul = any * pattern;
    auto any_pattern_mul2 = any * pattern;
    auto sum_mul_pattern = any_pattern_mul + any_pattern_mul2;
    //pattern->reset();

    auto any_pattern_mul3 = pattern * any;
    auto sum_mul_pattern2 = any_pattern_mul3 + any_pattern_mul2;
    std::cout << "FAILING TEST =============" << std::endl;
    std::cout << "a = " << a << std::endl;
    std::cout << "b = " << b << std::endl;
    std::cout << "c = " << c << std::endl;
    std::cout << " =========  " << std::endl;
    ASSERT_TRUE(n.match(sum_mul_pattern2, sumac));
    ASSERT_EQ(pattern->get_binded_node(), b);

    */

    //
    // 
    //{AT}
    //ASSERT_TRUE(n.match(ANY, A));
    
	
	//auto PATTERN3 = std::make_shared<pattern::op::Pattern>();
	//auto PATTERN_SUM = A + PATTERN3;
    //ASSERT_TRUE(n.match(PATTERN_SUM, SUM);

    //auto PATTERN_SUM_PERM = B + PATTERN;
    //ASSERT_TRUE(PATTERN_SUM_PERM, SUM);

    //auto PATTERN2 = std::make_shared<pattern::op::Pattern>();
    //auto ANYmulPATTERN2 = ANY*PATTERN2;
    //auto ANYmulPATTERN2_2 = ANY*PATTERN2;
    //A*B + C*B

    //auto AmulB = A * B;
    //auto BmulC = B * C;
    //auto SumOfMul = AmulB + BmulC;

    
    

    //TestMatcher test_matcher;
    /*ASSERT_TRUE(test_matcher.compare_nodes(A, B));
    ASSERT_FALSE(test_matcher.compare_nodes(A, SUM));
    ASSERT_TRUE(test_matcher.compare_nodes(ANY, A));
    ASSERT_TRUE(test_matcher.compare_nodes(ANY, B));
    ASSERT_TRUE(test_matcher.compare_nodes(ANY, SUM));*/
}



/*
TEST(pattern, op_op)
{
    auto shape = Shape{2, 2};
    auto A = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto B = make_shared<op::Parameter>(element::Float32::element_type(), shape);
    auto SUM = A + B;
    auto ANY = std::make_shared<pattern::op::Any>();

    TestMatcher test_matcher;
    ASSERT_TRUE(test_matcher.compare_nodes(A, B));
    ASSERT_FALSE(test_matcher.compare_nodes(A, SUM));
    ASSERT_TRUE(test_matcher.compare_nodes(ANY, A));
    ASSERT_TRUE(test_matcher.compare_nodes(ANY, B));
    ASSERT_TRUE(test_matcher.compare_nodes(ANY, SUM));
}
*/
