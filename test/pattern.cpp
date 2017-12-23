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
#include <cstdio>
#include <iostream>
#include <list>
#include <memory>

#include "gtest/gtest.h"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/topological_sort.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/any.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

//this is for more nuanced testing
class TestMatcher : public pattern::Matcher
{
    using pattern::Matcher::Matcher;
    bool virtual match_node(const std::shared_ptr<Node>& pattern_node,
                            const std::shared_ptr<Node>& graph_node,
                            PatternMap& pattern_map) override
    {
        if (std::dynamic_pointer_cast<::ngraph::op::Parameter>(pattern_node))
        {
            return pattern_node.get() == dynamic_cast<::ngraph::op::Parameter*>(graph_node.get());
        }

        return this->pattern::Matcher::match_node(pattern_node, graph_node, pattern_map);
    }

public:
    bool match(const std::shared_ptr<Node>& pattern_node, const std::shared_ptr<Node>& graph_node)
    {
        assert(
            pattern_node &&
            graph_node); //the same condition throws an exception in the non-test version of `match`
        NGRAPH_DEBUG << "Starting match pattern = " << pattern_node << " , "
                     << pattern_node->get_name() << " , graph_node = " << graph_node << " , "
                     << graph_node->get_name();

        m_pattern_map.clear();
        m_match_root.reset();

        bool is_match = match_node(pattern_node, graph_node, m_pattern_map);
        if (is_match)
        {
            m_match_root = graph_node;
        }
        return is_match;
    }
};

static std::shared_ptr<Node> construct_constant_node(int n)
{
    return op::Constant::create(element::i32, Shape{1}, {n});
}

class TestGraphRewrite : public ngraph::pass::GraphRewrite
{
public:
    void construct_multiply_by_one()
    {
        //pattern #1 : a * 1 = a
        auto iconst1 = construct_constant_node(1);
        auto pattern = pattern::op::Label::make_from_node(iconst1);

        NGRAPH_DEBUG << "IN TestGraphRewrite";

        ngraph::pattern::gr_callback_fn callback = [pattern](pattern::Matcher& m) {
            NGRAPH_DEBUG << "IN CALLBACK";
            assert(m.match_root()->get_input_ops().size() == 2);

            auto pattern_map = m.get_pattern_map();

            size_t const_node_index = m.match_root()->get_input_ops().at(0) == pattern_map[pattern];
            auto const_node = dynamic_pointer_cast<op::Constant>(
                m.match_root()->get_input_ops().at(const_node_index));
            auto second_node = m.match_root()->get_input_ops().at(const_node_index);
            NGRAPH_DEBUG << "second_node " << second_node->description() << " , " << second_node;
            NGRAPH_DEBUG << "pattern " << pattern_map[pattern]->description() << " , "
                         << pattern_map[pattern];
            ASSERT_TRUE(const_node);

            if (pattern_map[pattern]->get_element_type() != const_node->get_element_type() ||
                pattern_map[pattern]->get_shape() != const_node->get_shape())
            {
                NGRAPH_DEBUG << "TYPE/SHAPE";
                return;
            }

            auto const_values = const_node->get_vector<int32_t>();
            bool all_ones =
                std::all_of(begin(const_values), end(const_values), [](int e) { return e == 1; });

            if (!all_ones)
            {
                NGRAPH_DEBUG << "ALL_ONES";
                return;
            }

            NGRAPH_DEBUG << "BEFORE REPLACE";
            ngraph::replace_node(m.match_root(), pattern_map[pattern]);
        };

        auto m = make_shared<TestMatcher>(pattern * iconst1, callback);
        this->add_matcher(m);
    }

    void construct_add_zero()
    {
        //pattern #2 : a + 0 = a
        auto iconst0 = construct_constant_node(0);
        auto pattern = pattern::op::Label::make_from_node(iconst0);

        NGRAPH_DEBUG << "IN TestGraphRewrite";

        ngraph::pattern::gr_callback_fn callback = [pattern](pattern::Matcher& m) {
            NGRAPH_DEBUG << "IN CALLBACK";
            assert(m.match_root()->get_input_ops().size() == 2);

            auto pattern_map = m.get_pattern_map();

            size_t const_node_index = m.match_root()->get_input_ops().at(0) == pattern_map[pattern];
            auto const_node = dynamic_pointer_cast<op::Constant>(
                m.match_root()->get_input_ops().at(const_node_index));
            auto second_node = m.match_root()->get_input_ops().at(const_node_index);
            NGRAPH_DEBUG << "second_node " << second_node->description() << " , " << second_node;
            NGRAPH_DEBUG << "pattern " << pattern_map[pattern]->description() << " , "
                         << pattern_map[pattern];
            ASSERT_NE(nullptr, const_node);

            if (pattern_map[pattern]->get_element_type() != const_node->get_element_type() ||
                pattern_map[pattern]->get_shape() != const_node->get_shape())
            {
                NGRAPH_DEBUG << "TYPE/SHAPE";
                return;
            }

            auto const_values = const_node->get_vector<int>();
            bool all_zeros =
                std::all_of(begin(const_values), end(const_values), [](int e) { return e == 0; });

            if (!all_zeros)
            {
                NGRAPH_DEBUG << "ALL_ZEROS";
                return;
            }

            NGRAPH_DEBUG << "BEFORE REPLACE";
            ngraph::replace_node(m.match_root(), pattern_map[pattern]);
        };

        auto m = make_shared<TestMatcher>(pattern + iconst0, callback);
        this->add_matcher(m);
    }

    TestGraphRewrite()
        : GraphRewrite()
    {
        construct_multiply_by_one();
        construct_add_zero();
    }
};

static void run_passes(pass::Manager& pass_manager,
                       shared_ptr<Node> graph,
                       std::vector<shared_ptr<op::Parameter>> parms)
{
    auto shape = Shape{1};
    auto rt = make_shared<TensorViewType>(element::i32, shape);
    auto func = make_shared<Function>(graph, rt, op::Parameters{parms});
    pass_manager.run_passes(func);
}

TEST(pattern, graph_rewrite)
{
    auto shape = Shape{1};
    pass::Manager pass_manager;

    pass_manager.register_pass<pass::TopologicalSort>();
    pass_manager.register_pass<TestGraphRewrite>();

    {
        auto a = make_shared<op::Parameter>(element::i32, shape);
        auto b = make_shared<op::Parameter>(element::i32, shape);
        auto iconst0 = construct_constant_node(0);
        auto sum = (a + iconst0);
        auto graph = b + sum;
        run_passes(pass_manager, graph, {a, b});
        ASSERT_EQ(graph->get_input_ops().at(1), a);
        ASSERT_EQ(&graph->get_inputs().at(1).get_output(),
                  &a->get_outputs().at(0)); //graph's input points to a's output
        ASSERT_TRUE(sum->get_output_inputs(0)
                        .empty()); //graph's input is removed from sum's output.get_inputs()
        ASSERT_TRUE(a->get_outputs().at(0).get_inputs().count(
            &graph->get_inputs().at(1))); //a's output feeds into graph's input
    }

    {
        auto a = make_shared<op::Parameter>(element::i32, shape);
        auto b = make_shared<op::Parameter>(element::i32, shape);
        auto iconst1 = construct_constant_node(1);
        auto mul = (a * iconst1);
        auto graph = b + mul;
        run_passes(pass_manager, graph, {a, b});
        ASSERT_EQ(graph->get_input_ops().at(1), a);
        ASSERT_EQ(&graph->get_inputs().at(1).get_output(),
                  &a->get_outputs().at(0)); //graph's input points to a's output
        ASSERT_TRUE(mul->get_outputs()
                        .at(0)
                        .get_inputs()
                        .empty()); //graph's input is removed from sum's output.get_inputs()
        ASSERT_TRUE(a->get_outputs().at(0).get_inputs().count(
            &graph->get_inputs().at(1))); //a's output feeds into graph's input
    }

    {
        auto a = make_shared<op::Parameter>(element::i32, shape);
        auto b = make_shared<op::Parameter>(element::i32, shape);
        auto iconst1 = construct_constant_node(1);
        auto graph = ((((a * iconst1) * iconst1) * iconst1) * iconst1) + b;
        run_passes(pass_manager, graph, {a, b});
        ASSERT_EQ(graph->get_input_ops().at(0), a);
        ASSERT_EQ(&graph->get_inputs().at(0).get_output(),
                  &a->get_outputs().at(0)); //graph's input points to a's output
        ASSERT_TRUE(a->get_outputs().at(0).get_inputs().count(
            &graph->get_inputs().at(0))); //a's output feeds into graph's input
    }

    {
        auto a = make_shared<op::Parameter>(element::i32, shape);
        auto b = make_shared<op::Parameter>(element::i32, shape);
        auto iconst0 = construct_constant_node(0);
        auto iconst1 = construct_constant_node(1);
        auto graph = b + (iconst0 + ((a + iconst0) * iconst1));
        run_passes(pass_manager, graph, {a, b});
        ASSERT_EQ(graph->get_input_ops().at(1), a);
        ASSERT_EQ(&graph->get_inputs().at(1).get_output(),
                  &a->get_outputs().at(0)); //graph's input points to a's output
        ASSERT_TRUE(a->get_outputs().at(0).get_inputs().count(
            &graph->get_inputs().at(1))); //a's output feeds into graph's input
    }

    {
        auto a = make_shared<op::Parameter>(element::i32, shape);
        auto b = make_shared<op::Parameter>(element::i32, shape);
        auto iconst1 = construct_constant_node(1);
        auto graph = b + (iconst1 * (iconst1 * (iconst1 * (iconst1 * a))));
        run_passes(pass_manager, graph, {a, b});
        ASSERT_EQ(graph->get_input_ops().at(1), a);
        ASSERT_EQ(&graph->get_inputs().at(1).get_output(),
                  &a->get_outputs().at(0)); //graph's input points to a's output
        ASSERT_TRUE(a->get_outputs().at(0).get_inputs().count(
            &graph->get_inputs().at(1))); //a's output feeds into graph's input
    }
}

TEST(pattern, matcher)
{
    auto shape = Shape{1};

    auto a = make_shared<op::Parameter>(element::i32, shape);
    TestMatcher n(nullptr);
    ASSERT_TRUE(n.match(a, a));

    auto abs = make_shared<op::Abs>(a);
    auto any = std::make_shared<pattern::op::Any>(a);
    ASSERT_TRUE(n.match(any, abs));

    auto any_false =
        std::make_shared<pattern::op::Any>(a, [](std::shared_ptr<Node> no) { return false; });
    ASSERT_TRUE(n.match(any_false, a));

    auto pattern = pattern::op::Label::make_from_node(a);
    ASSERT_TRUE(n.match(pattern, a));
    ASSERT_EQ(n.get_pattern_map()[pattern], a);

    auto pattern_false =
        pattern::op::Label::make_from_node(a, [](std::shared_ptr<Node> no) { return false; });
    ASSERT_FALSE(n.match(pattern_false, a));

    auto b = make_shared<op::Parameter>(element::i32, shape);
    auto d = make_shared<op::Parameter>(element::i32, shape);
    ASSERT_FALSE(n.match(d, b));

    ASSERT_FALSE(n.match(abs + b, b + b));
    ASSERT_TRUE(n.match(any + b, abs + b));

    ASSERT_TRUE(n.match(pattern + b, abs + b));
    ASSERT_EQ(n.get_pattern_map()[pattern], abs);

    ASSERT_TRUE(n.match(b + pattern, abs + b));
    ASSERT_EQ(n.get_pattern_map()[pattern], abs);

    auto c = make_shared<op::Parameter>(element::i32, shape);
    ASSERT_TRUE(n.match(c * (b + pattern), c * (abs + b)));
    ASSERT_EQ(n.get_pattern_map()[pattern], abs);

    ASSERT_TRUE(n.match(c * (any + b), c * (abs + b)));     //nested any
    ASSERT_TRUE(n.match(c * (any + b), (b + abs) * c));     //permutations w/ any
    ASSERT_TRUE(n.match(c * (any_false + b), c * (a + b))); //nested any
    ASSERT_TRUE(n.match(c * (any_false + b), (b + a) * c)); //permutations w/ any_false

    auto iconst1_0 = construct_constant_node(1);
    auto iconst1_1 = construct_constant_node(1);
    ASSERT_TRUE(n.match(pattern * iconst1_0, a * iconst1_1)); //different iconst
    ASSERT_EQ(n.get_pattern_map()[pattern], a);
    auto fconst1_0 = op::Constant::create(element::f32, Shape{1}, {1});
    auto patternf = pattern::op::Label::make_from_node(fconst1_0);
    ASSERT_TRUE(n.match(patternf * fconst1_0, a * iconst1_1)); //different iconst

    //Subgraph labels
    auto add = a + b;
    auto label = pattern::op::Label::wrap(add);
    ASSERT_TRUE(n.match(label, add));
    ASSERT_EQ(n.get_pattern_map()[label], add);

    ASSERT_FALSE(n.match(label, a - b));

    ASSERT_TRUE(n.match(make_shared<op::Abs>(label), make_shared<op::Abs>(add)));
    ASSERT_EQ(n.get_pattern_map()[label], add);

    //Correlations
    auto label1 = pattern::op::Label::make_from_node(a);
    auto label2 = pattern::op::Label::wrap(label1 + b);
    auto sub_label1 = label1 - label2;
    ASSERT_TRUE(n.match(sub_label1, a - add));
    ASSERT_EQ(n.get_pattern_map()[label1], a);
    ASSERT_EQ(n.get_pattern_map()[label2], add);

    ASSERT_FALSE(n.match(sub_label1, add - a));

    auto add_label1 = label1 + label2;
    ASSERT_TRUE(n.match(add_label1, add + a));
    ASSERT_EQ(n.get_pattern_map()[label1], a);
    ASSERT_EQ(n.get_pattern_map()[label2], add);
}
