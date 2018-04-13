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
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/any.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/runtime/cpu/pass/cpu_fusion.hpp"
#include "ngraph/serializer.hpp"
#include "util/matcher.hpp"
#include "util/test_tools.hpp"

using namespace ngraph;
using namespace std;

template <typename T>
std::shared_ptr<Node> create_reduction(const std::shared_ptr<Node>& node,
                                       const std::string& init_val,
                                       const AxisSet& reduction_axes)
{
    const auto& et = node->get_element_type();
    auto f_A = std::make_shared<op::Parameter>(et, Shape{});
    auto f_B = std::make_shared<op::Parameter>(et, Shape{});
    auto f =
        std::make_shared<Function>(std::make_shared<T>(f_A, f_B), op::ParameterVector{f_A, f_B});

    auto init = std::make_shared<op::Constant>(et, Shape{}, std::vector<std::string>({init_val}));
    return std::make_shared<op::Reduce>(node, init, f, reduction_axes);
}

std::shared_ptr<Node> xla_sum(const std::shared_ptr<Node>& node, const AxisSet& reduction_axes)
{
    return create_reduction<op::Add>(node, "0", reduction_axes);
}

static std::shared_ptr<Node> construct_constant_node(int n)
{
    return op::Constant::create(element::i32, Shape{}, {n});
}

bool is_zero(std::shared_ptr<Node> reduce_constant)
{
    return is_equal_to_const_value("0", reduce_constant);
}

bool sum_predicate(std::shared_ptr<Node> gn)
{
    NGRAPH_DEBUG << "pred_v2 : looking at " << gn->get_name();
    if (auto r = std::dynamic_pointer_cast<op::Reduce>(gn))
    {
        auto reducee = gn->get_argument(0);
        auto reduce_constant = gn->get_argument(1);

        if (!is_zero(reduce_constant))
        {
            return false;
        }

        auto result = r->get_functions()[0]->get_result()->get_argument(0);
        NGRAPH_DEBUG << "looking at function's result  " << result->get_name();
        if (auto sum = std::dynamic_pointer_cast<op::Add>(result))
        {
            auto parm1 = std::dynamic_pointer_cast<op::Parameter>(sum->get_argument(0));
            auto parm2 = std::dynamic_pointer_cast<op::Parameter>(sum->get_argument(1));

            const auto parm_or_nil = [](std::shared_ptr<Node> p) {
                return p ? p->get_name() : std::string("(nil)");
            };
            NGRAPH_DEBUG << "parm1 = " << parm_or_nil(parm1) << " , parm2 = " << parm_or_nil(parm2)
                         << std::endl;
            if (parm1 && parm2 && parm1 != parm2)
            {
                return true;
            }
        }
    }

    return false;
}

std::shared_ptr<pattern::op::Label> construct_sum_pattern() //for the sake of explicitness
{
    return std::make_shared<pattern::op::Label>(element::i32, Shape{}, sum_predicate);
}

static std::shared_ptr<pattern::op::Label> construct_variance_graph()
{
    // construct varaiance
    auto N = op::Constant::create(element::f32, Shape{3}, {2, 2, 2});
    auto input = std::make_shared<pattern::op::Label>(element::f32, Shape{2, 3});
    auto input_sq = std::make_shared<op::Multiply>(input, input);
    auto sum_input = std::make_shared<op::Sum>(input, AxisSet{0});
    auto square_sumed_input = std::make_shared<op::Multiply>(sum_input, sum_input);
    auto sum_squared_input = std::make_shared<op::Sum>(input_sq, AxisSet{0});
    auto avg_input_sum_sq = std::make_shared<op::Divide>(square_sumed_input, N);
    auto xmu = std::make_shared<op::Subtract>(sum_squared_input, avg_input_sum_sq);
    auto variance = std::make_shared<op::Divide>(xmu, N);
    auto variance_label =
        std::make_shared<pattern::op::Label>(variance, nullptr, NodeVector{variance});

    return variance_label;
}

static std::shared_ptr<pattern::op::Label> construct_mean_graph()
{
    //construct mean;
    auto input = std::make_shared<pattern::op::Label>(element::f32, Shape{2, 3});
    auto N = op::Constant::create(element::f32, Shape{3}, {2, 2, 2});
    auto sum_input1 = std::make_shared<op::Sum>(input, AxisSet{0});
    auto mean = std::make_shared<op::Divide>(sum_input1, N);
    auto mean_label = std::make_shared<pattern::op::Label>(mean, nullptr, NodeVector{mean});
    return mean_label;
}

class TestGraphRewrite : public ngraph::pass::GraphRewrite
{
public:
    void construct_multiply_by_one()
    {
        //pattern #1 : a * 1 = a
        auto iconst1 = construct_constant_node(1);
        auto pattern = std::make_shared<pattern::op::Label>(iconst1);

        ngraph::pattern::graph_rewrite_callback callback = [pattern](pattern::Matcher& m) {
            NGRAPH_DEBUG << "In a callback for construct_multiply_by_one against "
                         << m.match_root()->get_name();
            assert(m.match_root()->get_arguments().size() == 2);

            auto pattern_map = m.get_pattern_map();

            size_t const_node_index = m.match_root()->get_arguments().at(0) == pattern_map[pattern];
            auto const_node = dynamic_pointer_cast<op::Constant>(
                m.match_root()->get_arguments().at(const_node_index));
            auto second_node = m.match_root()->get_arguments().at(const_node_index);
            NGRAPH_DEBUG << "second_node = " << second_node->get_name()
                         << " , pattern = " << pattern_map[pattern]->get_name();

            if (pattern_map[pattern]->get_element_type() != const_node->get_element_type() ||
                pattern_map[pattern]->get_shape() != const_node->get_shape())
            {
                NGRAPH_DEBUG << "Operands' types and/or shape don't match";
                return false;
            }

            auto const_values = const_node->get_vector<int32_t>();
            bool all_ones =
                std::all_of(begin(const_values), end(const_values), [](int e) { return e == 1; });

            if (!all_ones)
            {
                NGRAPH_DEBUG << "Constant vector's values aren't equal to 1";
                return false;
            }

            ngraph::replace_node(m.match_root(), pattern_map[pattern]);
            return true;
        };

        auto m = make_shared<TestMatcher>(pattern * iconst1, callback);
        this->add_matcher(m);
    }

    void construct_add_zero()
    {
        //pattern #2 : a + 0 = a
        auto iconst0 = construct_constant_node(0);
        auto pattern = std::make_shared<pattern::op::Label>(iconst0);

        auto callback = [pattern](pattern::Matcher& m) {
            NGRAPH_DEBUG << "In a callback for construct_add_zero against "
                         << m.match_root()->get_name();
            assert(m.match_root()->get_arguments().size() == 2);

            auto pattern_map = m.get_pattern_map();

            size_t const_node_index = m.match_root()->get_arguments().at(0) == pattern_map[pattern];
            auto const_node = dynamic_pointer_cast<op::Constant>(
                m.match_root()->get_arguments().at(const_node_index));
            auto second_node = m.match_root()->get_arguments().at(const_node_index);
            NGRAPH_DEBUG << "second_node = " << second_node->get_name()
                         << " , pattern = " << pattern_map[pattern]->get_name();

            if (pattern_map[pattern]->get_element_type() != const_node->get_element_type() ||
                pattern_map[pattern]->get_shape() != const_node->get_shape())
            {
                NGRAPH_DEBUG << "Operands' types and/or shape don't match";
                return false;
            }

            auto const_values = const_node->get_vector<int>();
            bool all_zeros =
                std::all_of(begin(const_values), end(const_values), [](int e) { return e == 0; });

            if (!all_zeros)
            {
                NGRAPH_DEBUG << "Constant vector's values aren't equal to 0";
                return false;
            }

            ngraph::replace_node(m.match_root(), pattern_map[pattern]);
            return true;
        };

        auto m = make_shared<TestMatcher>(pattern + iconst0, callback);
        this->add_matcher(m);
    }

    void construct_sum()
    {
        auto sum_pattern = construct_sum_pattern();

        ngraph::pattern::graph_rewrite_callback callback = [](pattern::Matcher& m) {
            NGRAPH_DEBUG << "In a callback for construct_sum_pattern against "
                         << m.match_root()->get_name();
            auto reduce = std::dynamic_pointer_cast<op::Reduce>(m.match_root());
            auto reducee = reduce->get_inputs().at(0).get_output().get_node();
            NGRAPH_DEBUG << "reducee = " << reducee->get_name();
            auto sum =
                std::shared_ptr<ngraph::Node>(new op::Sum(reducee, reduce->get_reduction_axes()));

            ngraph::replace_node(m.match_root(), sum);
            return true;
        };

        auto m = make_shared<TestMatcher>(sum_pattern, callback);
        this->add_matcher(m);
    }

    TestGraphRewrite()
        : GraphRewrite()
    {
        construct_multiply_by_one();
        construct_add_zero();
        construct_sum();
    }
};

static void run_passes(pass::Manager& pass_manager,
                       shared_ptr<Node> graph,
                       std::vector<shared_ptr<op::Parameter>> parms)
{
    auto func = make_shared<Function>(graph, op::ParameterVector{parms});
    pass_manager.run_passes(func);
}

TEST(pattern, graph_rewrite)
{
    Shape shape{};
    pass::Manager pass_manager;
    pass_manager.register_pass<TestGraphRewrite>();

    {
        auto a = make_shared<op::Parameter>(element::i32, shape);
        auto b = make_shared<op::Parameter>(element::i32, shape);
        auto c = make_shared<op::Parameter>(element::i32, shape);
        auto iconst0 = construct_constant_node(0);
        auto graph_a = a + iconst0;
        auto graph_b = b + iconst0;

        auto f = std::make_shared<Function>(ngraph::NodeVector{a, b, graph_a, c, graph_b},
                                            op::ParameterVector{a, b, c});
        pass_manager.run_passes(f);

        ASSERT_TRUE(graph_a->get_output_inputs(0).empty());
        ASSERT_TRUE(graph_b->get_output_inputs(0).empty());

        auto expected = ngraph::NodeVector{a, b, a, c, b};
        ASSERT_TRUE(count_ops_of_type<op::Add>(f) == 0);
    }

    {
        auto a = make_shared<op::Parameter>(element::i32, shape);
        auto b = make_shared<op::Parameter>(element::i32, shape);
        auto iconst0 = construct_constant_node(0);
        auto sum = (a + iconst0);
        auto graph = b + sum;
        run_passes(pass_manager, graph, {a, b});
        ASSERT_EQ(graph->get_arguments().at(1), a);
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
        ASSERT_EQ(graph->get_arguments().at(1), a);
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
        ASSERT_EQ(graph->get_arguments().at(0), a);
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
        ASSERT_EQ(graph->get_arguments().at(1), a);
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
        ASSERT_EQ(graph->get_arguments().at(1), a);
        ASSERT_EQ(&graph->get_inputs().at(1).get_output(),
                  &a->get_outputs().at(0)); //graph's input points to a's output
        ASSERT_TRUE(a->get_outputs().at(0).get_inputs().count(
            &graph->get_inputs().at(1))); //a's output feeds into graph's input
    }

    //Sum rewrite
    {
        auto parm = make_shared<op::Parameter>(element::i32, Shape{2, 2});
        auto axes = AxisSet{0, 1};
        auto sum_graph = xla_sum(parm, axes);
        auto innermost_abs = make_shared<op::Abs>(sum_graph);

        auto nested_sum_graph = make_shared<op::Abs>(
            make_shared<op::Abs>(make_shared<op::Abs>(make_shared<op::Abs>(innermost_abs))));

        run_passes(pass_manager, nested_sum_graph, {parm});
        auto sum = std::dynamic_pointer_cast<op::Sum>(innermost_abs->get_argument(0));
        ASSERT_TRUE(sum);
        ASSERT_EQ(sum->get_reduction_axes(), axes);
        ASSERT_EQ(sum->get_argument(0), parm);
    }
}

TEST(pattern, matcher)
{
    Shape shape{};
    auto a = make_shared<op::Parameter>(element::i32, shape);
    TestMatcher n(nullptr);
    ASSERT_TRUE(n.match(a, a));

    auto abs = make_shared<op::Abs>(a);
    auto any = std::make_shared<pattern::op::Any>(a);
    ASSERT_TRUE(n.match(any, abs));

    auto any_false =
        std::make_shared<pattern::op::Any>(a, [](std::shared_ptr<Node> no) { return false; });
    ASSERT_TRUE(n.match(any_false, a));

    auto pattern = std::make_shared<pattern::op::Label>(a);
    ASSERT_TRUE(n.match(pattern, a));
    ASSERT_EQ(n.get_pattern_map()[pattern], a);

    auto pattern_false =
        std::make_shared<pattern::op::Label>(a, [](std::shared_ptr<Node> no) { return false; });
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
    auto fconst1_0 = op::Constant::create(element::f32, shape, {1});
    auto patternf = std::make_shared<pattern::op::Label>(fconst1_0);
    ASSERT_TRUE(n.match(patternf * fconst1_0, a * iconst1_1)); //different iconst

    //Subgraph labels
    auto add = a + b;
    auto label = std::make_shared<pattern::op::Label>(add, nullptr, NodeVector{add});
    ASSERT_TRUE(n.match(label, add));
    ASSERT_EQ(n.get_pattern_map()[label], add);

    ASSERT_FALSE(n.match(label, a - b));

    ASSERT_TRUE(n.match(make_shared<op::Abs>(label), make_shared<op::Abs>(add)));
    ASSERT_EQ(n.get_pattern_map()[label], add);

    //Correct argument order
    ASSERT_FALSE(n.match(b - a, a - b));
    auto aab = a * (a - b);
    auto paab = pattern * (pattern - b);
    ASSERT_TRUE(n.match(paab, aab));
    auto aba = a * (b - a);
    ASSERT_FALSE(n.match(paab, aba));
    auto paba = pattern * (b - pattern);
    ASSERT_FALSE(n.match(paba, aab));

    //Correlations
    auto label1 = std::make_shared<pattern::op::Label>(a);
    auto tmp = label1 + b;
    auto label2 = std::make_shared<pattern::op::Label>(tmp, nullptr, NodeVector{tmp});
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

TEST(pattern, sum)
{
    //Sum
    TestMatcher n(nullptr);
    auto reducee_const = std::make_shared<op::Constant>(
        element::i32, Shape{2, 2}, std::vector<std::string>({"0", "0", "0", "0"}));
    auto sum_graph = xla_sum(reducee_const, AxisSet{0, 1});

    auto reduce_label = construct_sum_pattern();
    ASSERT_TRUE(n.match(reduce_label, sum_graph));
    ASSERT_EQ(n.get_pattern_map()[reduce_label], sum_graph);

    auto nested_sum_graph = make_shared<op::Abs>(make_shared<op::Abs>(
        make_shared<op::Abs>(make_shared<op::Abs>(make_shared<op::Abs>(sum_graph)))));

    auto nested_reduce_label = make_shared<op::Abs>(make_shared<op::Abs>(
        make_shared<op::Abs>(make_shared<op::Abs>(make_shared<op::Abs>(reduce_label)))));

    ASSERT_TRUE(n.match(nested_reduce_label, nested_sum_graph));
    ASSERT_EQ(n.get_pattern_map()[reduce_label], sum_graph);
}

TEST(pattern, mean)
{
    //construct mean
    TestMatcher n(nullptr);

    auto input = std::make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto N = op::Constant::create(element::f32, Shape{3}, {2, 2, 2});
    auto sum_input1 = std::make_shared<op::Sum>(input, AxisSet{0});
    auto mean = std::make_shared<op::Divide>(sum_input1, N);

    auto mean_graph = construct_mean_graph();
    ASSERT_TRUE(n.match(mean_graph, mean));
    ASSERT_EQ(n.get_pattern_map()[mean_graph], mean);
}

TEST(pattern, variance)
{
    //construct variance
    TestMatcher n(nullptr);
    auto N = op::Constant::create(element::f32, Shape{3}, {2, 2, 2});
    auto input = std::make_shared<pattern::op::Label>(element::f32, Shape{2, 3});
    auto input_sq = std::make_shared<op::Multiply>(input, input);
    auto sum_input = std::make_shared<op::Sum>(input, AxisSet{0});
    auto square_sumed_input = std::make_shared<op::Multiply>(sum_input, sum_input);
    auto sum_squared_input = std::make_shared<op::Sum>(input_sq, AxisSet{0});
    auto avg_input_sum_sq = std::make_shared<op::Divide>(square_sumed_input, N);
    auto xmu = std::make_shared<op::Subtract>(sum_squared_input, avg_input_sum_sq);
    auto variance = std::make_shared<op::Divide>(xmu, N);

    auto var_graph = construct_variance_graph();
    ASSERT_TRUE(n.match(var_graph, variance));
    ASSERT_EQ(n.get_pattern_map()[var_graph], variance);
}

TEST(pattern, previous_matches)
{
    using ngraph::pattern::Matcher;
    Shape shape{};
    Matcher::PatternMap previous_matches;
    auto a = make_shared<op::Parameter>(element::i32, shape);
    auto b = make_shared<op::Parameter>(element::i32, shape);
    auto pattern = std::make_shared<pattern::op::Label>(b);
    auto abs = make_shared<op::Abs>(a);
    auto add = abs + b;
    {
        Matcher n(pattern + b);
        ASSERT_TRUE(n.match(add, previous_matches));
        ASSERT_EQ(n.get_pattern_map()[pattern], abs);
    }

    {
        Matcher n(pattern + b);
        previous_matches.insert(std::make_pair(pattern, a));
        ASSERT_FALSE(n.match(add, previous_matches));
    }
}

TEST(pattern, recurrent_pattern)
{
    using ngraph::pattern::RecurrentMatcher;
    Shape shape{};
    ngraph::pattern::Matcher::PatternMap previous_matches;
    auto a = make_shared<op::Parameter>(element::i32, shape);
    auto b = make_shared<op::Parameter>(element::i32, shape);
    auto rpattern = std::make_shared<pattern::op::Label>(b);
    auto iconst0 = construct_constant_node(0);
    auto abs = make_shared<op::Abs>(a);
    auto add1 = iconst0 + b;
    auto add2 = iconst0 + add1;
    auto add3 = iconst0 + add2;
    auto padd = iconst0 + rpattern;
    std::set<std::shared_ptr<pattern::op::Label>> empty_correlated_matches;
    RecurrentMatcher rm(padd, rpattern, empty_correlated_matches, nullptr);
    ASSERT_TRUE(rm.match(add3));
    ASSERT_EQ(rm.get_number_of_bound_labels(), 1);
    auto recurrent_matches = rm.get_bound_nodes_for_pattern(rpattern);
    ASSERT_EQ(recurrent_matches.at(0), add2);
    ASSERT_EQ(recurrent_matches.at(1), add1);
    ASSERT_EQ(recurrent_matches.at(2), b);

    //Multiple labels in a reccuring pattern
    auto iconst1 = construct_constant_node(1);
    auto iconst_label = std::make_shared<pattern::op::Label>(iconst1, nullptr, NodeVector{iconst1});
    auto add2_2 = iconst1 + add1;
    auto add3_2 = iconst0 + add2_2;
    auto padd2 = iconst_label + rpattern;
    RecurrentMatcher rm2(padd2, rpattern, empty_correlated_matches, nullptr);
    ASSERT_TRUE(rm2.match(add3_2));
    ASSERT_EQ(rm2.get_number_of_bound_labels(), 2);
    recurrent_matches = rm2.get_bound_nodes_for_pattern(rpattern);
    ASSERT_EQ(recurrent_matches.at(0), add2_2);
    ASSERT_EQ(recurrent_matches.at(1), add1);
    ASSERT_EQ(recurrent_matches.at(2), b);
    auto iconst_matches = rm2.get_bound_nodes_for_pattern(iconst_label);
    ASSERT_EQ(iconst_matches.at(0), iconst0);
    ASSERT_EQ(iconst_matches.at(1), iconst1);
    ASSERT_EQ(iconst_matches.at(2), iconst0);

    //Non-matching correlated labels
    std::set<std::shared_ptr<pattern::op::Label>> correlated_matches;
    correlated_matches.insert(iconst_label);
    RecurrentMatcher rm3(padd2, rpattern, correlated_matches, nullptr);
    ASSERT_TRUE(rm3.match(add3_2));
    ASSERT_EQ(rm3.get_number_of_bound_labels(), 2);
    iconst_matches = rm3.get_bound_nodes_for_pattern(iconst_label);
    ASSERT_EQ(iconst_matches.size(), 1);
    ASSERT_EQ(iconst_matches.at(0), iconst0);

    //Matching correlated labels and
    //testing if RecurrentMatcher can be reused for different nodes
    ASSERT_TRUE(rm3.match(add3));
    ASSERT_EQ(rm3.get_number_of_bound_labels(), 2);
    recurrent_matches = rm3.get_bound_nodes_for_pattern(rpattern);
    ASSERT_EQ(recurrent_matches.at(0), add2);
    ASSERT_EQ(recurrent_matches.at(1), add1);
    ASSERT_EQ(recurrent_matches.at(2), b);
    iconst_matches = rm3.get_bound_nodes_for_pattern(iconst_label);
    ASSERT_EQ(iconst_matches.at(0), iconst0);
    ASSERT_EQ(iconst_matches.at(1), iconst0);
    ASSERT_EQ(iconst_matches.at(2), iconst0);
}

class TestRecurrentGraphRewrite : public ngraph::pass::RecurrentGraphRewrite
{
public:
    void construct_recurrent_add()
    {
        Shape shape{};
        auto iconst0 = construct_constant_node(0);
        auto iconst_label =
            std::make_shared<pattern::op::Label>(iconst0, nullptr, NodeVector{iconst0});
        auto rpattern = std::make_shared<pattern::op::Label>(element::i32, shape);
        auto padd = iconst_label + rpattern;

        auto sum_pattern = construct_sum_pattern();

        ngraph::pattern::recurrent_graph_rewrite_callback callback = [iconst_label, rpattern](
            pattern::RecurrentMatcher& rm) {
            NGRAPH_DEBUG << "In a callback for construct_recurrent_add against "
                         << rm.get_match_root()->get_name();

            auto iconst_matches = rm.get_bound_nodes_for_pattern(iconst_label);

            auto is_iconst_zero = [](std::shared_ptr<Node> n) {
                bool result = is_zero(n);
                NGRAPH_DEBUG << n->get_name() << " is " << (result ? " a zero " : " not a zero");
                return is_zero(n);
            };

            bool are_all_iconst_zeros =
                std::all_of(iconst_matches.begin(), iconst_matches.end(), is_iconst_zero);

            if (!are_all_iconst_zeros)
            {
                return false;
            }

            auto number_of_adds = rm.get_number_of_recurrent_matches();
            //replace the topmost add with the seed (i.e. the first parameter to add)
            //matches are added in reverse order (i.e. the first match is the topmost node)
            auto arg = rm.get_bound_nodes_for_pattern(rpattern).at(number_of_adds - 1);
            NGRAPH_DEBUG << "Replacing " << rm.get_match_root()->get_name() << " with "
                         << arg->get_name();
            ngraph::replace_node(rm.get_match_root(), arg);
            return true;
        };

        std::set<std::shared_ptr<pattern::op::Label>> empty_correlated_matches;
        auto rm = make_shared<pattern::RecurrentMatcher>(
            padd, rpattern, empty_correlated_matches, callback);
        this->add_matcher(rm);
    }

    TestRecurrentGraphRewrite()
        : RecurrentGraphRewrite()
    {
        construct_recurrent_add();
    }
};

TEST(pattern, recurrent_graph_rewrite)
{
    Shape shape{};
    pass::Manager pass_manager;
    pass_manager.register_pass<TestRecurrentGraphRewrite>();

    {
        auto a = make_shared<op::Parameter>(element::i32, shape);
        auto iconst0 = construct_constant_node(0);
        auto add_a1 = a + iconst0;
        auto add_a2 = add_a1 + iconst0;
        auto add_a3 = add_a2 + iconst0;
        auto abs_add_a3 = std::make_shared<op::Abs>(add_a3);

        auto b = make_shared<op::Parameter>(element::i32, shape);
        auto add_b1 = b + iconst0;
        auto add_b2 = add_b1 + iconst0;
        auto abs_add_b2 = std::make_shared<op::Abs>(add_b2);

        auto graph = abs_add_a3 * abs_add_b2;

        auto f = std::make_shared<Function>(ngraph::NodeVector{graph}, op::ParameterVector{a, b});
        pass_manager.run_passes(f);

        auto left_abs = graph->get_argument(0);
        auto add_a = left_abs->get_argument(0);
        ASSERT_EQ(add_a, a);

        auto right_abs = graph->get_argument(1);
        auto add_b = right_abs->get_argument(0);
        ASSERT_EQ(add_b, b);
    }
}
