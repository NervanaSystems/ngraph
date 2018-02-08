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
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/ops/sum.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/any.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "util/matcher.hpp"
#include "ngraph/ops/sum.hpp"
#include "ngraph/ops/divide.hpp"
#include "ngraph/ops/multiply.hpp"
#include "ngraph/ops/subtract.hpp"
#include "ngraph/ops/add.hpp"
#include "ngraph/ops/sqrt.hpp"
#include "ngraph/ops/constant.hpp"
#include "ngraph/ops/batchnorm.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/runtime/cpu/pass/cpu_fusion.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/pass/reshape_elimination.hpp"

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
    auto f = std::make_shared<Function>(std::make_shared<T>(f_A, f_B), op::Parameters{f_A, f_B});

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

bool is_equal_to_const_value(std::string const_value, std::shared_ptr<Node> reduce_constant)
{
    if (auto rc = std::dynamic_pointer_cast<op::Constant>(reduce_constant))
    {
        auto cshape = rc->get_shape();
        size_t n = shape_size(cshape);
        //awkward(but generic) way to construct a constant of a given type, shape, value
        std::vector<std::string> vz{n, const_value};
        auto zero_constant = std::make_shared<op::Constant>(rc->get_element_type(), cshape, vz);

        //equally awkward way to compare elements to const_value
        size_t n_bytes = n * rc->get_element_type().size();
        NGRAPH_DEBUG << "Comparing " << n_bytes << " bytes";
        return !memcmp(zero_constant->get_data_ptr(), rc->get_data_ptr(), n_bytes);
    }
    else
    {
        return false;
    }
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
        auto reducee = gn->get_input_op(0);
        auto reduce_constant = gn->get_input_op(1);

        if (!is_zero(reduce_constant))
        {
            return false;
        }

        NGRAPH_DEBUG << "looking at function's result  "
                     << r->get_functions()[0]->get_result()->get_name();
        if (auto sum = std::dynamic_pointer_cast<op::Add>(r->get_functions()[0]->get_result()))
        {
            auto parm1 = std::dynamic_pointer_cast<op::Parameter>(sum->get_input_op(0));
            auto parm2 = std::dynamic_pointer_cast<op::Parameter>(sum->get_input_op(1));

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

static std::shared_ptr<pattern::op::Label> construct_variance_graph(){
    // construct varaiance
    auto N = op::Constant::create(element::f32, Shape{3}, {2, 2, 2});
    auto input = std::make_shared<pattern::op::Label>(element::f32, Shape{2, 3});
    auto input_sq = std::make_shared<op::Multiply>(input, input);
    auto sum_input = std::make_shared<op::Sum>(input, AxisSet{0});
    auto square_sumed_input = std::make_shared<op::Multiply>(sum_input, sum_input);
    auto sum_squared_input = std::make_shared<op::Sum>(input_sq, AxisSet{0});
    auto avg_input_sum_sq = std::make_shared<op::Divide>(square_sumed_input, N);
    auto xmu = std::make_shared<op::Subtract>(sum_squared_input, avg_input_sum_sq);
    auto variance  = std::make_shared<op::Divide>(xmu, N);
    auto variance_label = std::make_shared<pattern::op::Label>(variance, nullptr, Nodes{variance});

    return variance_label;
}

static std::shared_ptr<pattern::op::Label> construct_mean_graph(){
    //construct mean;
    auto input = std::make_shared<pattern::op::Label>(element::f32, Shape{2, 3});
    auto N = op::Constant::create(element::f32, Shape{3}, {2, 2, 2});
    auto sum_input1 = std::make_shared<op::Sum>(input, AxisSet{0});
    auto mean = std::make_shared<op::Divide>(sum_input1, N);
    auto mean_label = std::make_shared<pattern::op::Label>(mean, nullptr, Nodes{mean});
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

        ngraph::pattern::gr_callback_fn callback = [pattern](pattern::Matcher& m) {
            NGRAPH_DEBUG << "In a callback for construct_multiply_by_one against "
                         << m.match_root()->get_name();
            assert(m.match_root()->get_input_ops().size() == 2);

            auto pattern_map = m.get_pattern_map();

            size_t const_node_index = m.match_root()->get_input_ops().at(0) == pattern_map[pattern];
            auto const_node = dynamic_pointer_cast<op::Constant>(
                m.match_root()->get_input_ops().at(const_node_index));
            auto second_node = m.match_root()->get_input_ops().at(const_node_index);
            NGRAPH_DEBUG << "second_node = " << second_node->get_name()
                         << " , pattern = " << pattern_map[pattern]->get_name();

            std::shared_ptr<ngraph::Node> nn = nullptr;
            if (pattern_map[pattern]->get_element_type() != const_node->get_element_type() ||
                pattern_map[pattern]->get_shape() != const_node->get_shape())
            {
                NGRAPH_DEBUG << "Operands' types and/or shape don't match";
                return nn;
            }

            auto const_values = const_node->get_vector<int32_t>();
            bool all_ones =
                std::all_of(begin(const_values), end(const_values), [](int e) { return e == 1; });

            if (!all_ones)
            {
                NGRAPH_DEBUG << "Constant vector's values aren't equal to 1";
                return nn;
            }
            return pattern_map[pattern];
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
            assert(m.match_root()->get_input_ops().size() == 2);

            auto pattern_map = m.get_pattern_map();

            size_t const_node_index = m.match_root()->get_input_ops().at(0) == pattern_map[pattern];
            auto const_node = dynamic_pointer_cast<op::Constant>(
                m.match_root()->get_input_ops().at(const_node_index));
            auto second_node = m.match_root()->get_input_ops().at(const_node_index);
            NGRAPH_DEBUG << "second_node = " << second_node->get_name()
                         << " , pattern = " << pattern_map[pattern]->get_name();

            //ASSERT_NE(nullptr, const_node);

            std::shared_ptr<ngraph::Node> nn = nullptr;
            if (pattern_map[pattern]->get_element_type() != const_node->get_element_type() ||
                pattern_map[pattern]->get_shape() != const_node->get_shape())
            {
                NGRAPH_DEBUG << "Operands' types and/or shape don't match";
                return nn;
            }

            auto const_values = const_node->get_vector<int>();
            bool all_zeros =
                std::all_of(begin(const_values), end(const_values), [](int e) { return e == 0; });

            if (!all_zeros)
            {
                NGRAPH_DEBUG << "Constant vector's values aren't equal to 0";
                return nn;
            }

            return pattern_map[pattern];
        };

        auto m = make_shared<TestMatcher>(pattern + iconst0, callback);
        this->add_matcher(m);
    }

    void construct_sum()
    {
        auto sum_pattern = construct_sum_pattern();

        ngraph::pattern::gr_callback_fn callback = [](pattern::Matcher& m) {
            NGRAPH_DEBUG << "In a callback for construct_sum_pattern against "
                         << m.match_root()->get_name();
            auto reduce = std::dynamic_pointer_cast<op::Reduce>(m.match_root());
            auto reducee = reduce->get_inputs().at(0).get_output().get_node();
            NGRAPH_DEBUG << "reducee = " << reducee->get_name();
            auto sum =
                std::shared_ptr<ngraph::Node>(new op::Sum(reducee, reduce->get_reduction_axes()));
            return sum;
        };

        auto m = make_shared<TestMatcher>(sum_pattern, callback);
        this->add_matcher(m);
    }

    void construct_fprop_bn()
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
        auto variance  = std::make_shared<op::Divide>(xmu, N);
        auto variance_label = std::make_shared<pattern::op::Label>(variance, nullptr, Nodes{variance});
        auto variance_with_broadcast = std::make_shared<op::Broadcast>(variance_label, Shape{2, 3}, AxisSet{0});

        // construct mean
        auto sum_input1 = std::make_shared<op::Sum>(input, AxisSet{0});
        auto mean = std::make_shared<op::Divide>(sum_input1, N);
        auto mean_label = std::make_shared<pattern::op::Label>(mean, nullptr, Nodes{mean});
        auto mean_with_broadcast = std::make_shared<op::Broadcast>(mean_label, Shape{2, 3}, AxisSet{0});
        auto input_diff_mean = std::make_shared<op::Subtract>(input, mean_with_broadcast);

        // Eps
        auto eps_label = std::make_shared<pattern::op::Label>(element::f32, Shape{3});
        auto eps_with_broadcast = std::make_shared<op::Broadcast>(eps_label, Shape{2, 3}, AxisSet{0});

        auto add1 = std::make_shared<op::Add>(eps_with_broadcast, variance_with_broadcast);
        auto sqrt_variance_eps = std::make_shared<op::Sqrt>(add1);
        auto divide_mean_variance = std::make_shared<op::Divide>(input_diff_mean, sqrt_variance_eps);

        //Gamma
        auto gamma_label = std::make_shared<pattern::op::Label>(element::f32, Shape{3});
        auto gamma_with_broadcast = std::make_shared<op::Broadcast>(gamma_label, Shape{2, 3}, AxisSet{0});
        auto multiply_gamma =  std::make_shared<op::Multiply>(gamma_with_broadcast, divide_mean_variance);
        
        //Beta
        auto beta_label = std::make_shared<pattern::op::Label>(element::f32, Shape{3});
        auto beta_with_broadcast = std::make_shared<op::Broadcast>(beta_label, Shape{2, 3}, AxisSet{0});

        auto add_beta =  std::make_shared<op::Add>(beta_with_broadcast, multiply_gamma);
        // This completes fprop bn pattern

        //Define a call back that needs to called once the DFG matches the pattern 
        ngraph::pattern::gr_callback_fn callback = [variance_label, mean_label, input, eps_label, gamma_label, beta_label](pattern::Matcher& m) {
            NGRAPH_DEBUG << "In a callback for construct_fprop_bn pattern against "
                         << m.match_root()->get_name();

            std::shared_ptr<Node> nn = nullptr;
            auto pattern_map = m.get_pattern_map();
            NGRAPH_DEBUG << "Input: " << pattern_map[input]->get_name() << " " << pattern_map[input]->get_shape().size();
            NGRAPH_DEBUG << "Variance: " << pattern_map[variance_label]->get_name() << " " << pattern_map[variance_label]->get_shape().size();
            NGRAPH_DEBUG << "Mean: "  <<  pattern_map[mean_label]->get_name() << " " <<  pattern_map[mean_label]->get_shape().size();
            NGRAPH_DEBUG << "eps: " << pattern_map[eps_label]->get_name() << " " <<  pattern_map[eps_label]->get_shape().size();
            NGRAPH_DEBUG << "gamma: " << pattern_map[gamma_label]->get_name() << " " <<  pattern_map[gamma_label]->get_shape().size();
            NGRAPH_DEBUG << "beta: " << pattern_map[beta_label]->get_name() << " " <<  pattern_map[beta_label]->get_shape().size();

            // dont fuse if the inout doesnt have 4dims 
            if (pattern_map[input]->get_shape().size()!=4)
            {
                 NGRAPH_DEBUG << "Input to bn doesnt not have 4dims, so not fusing";
                 return nn;
             }
            Shape bn_output_shape{m.match_root()->get_shape()};
            Shape bn_mean_shape{pattern_map[mean_label]->get_shape()};
            Shape bn_variance_shape{pattern_map[variance_label]->get_shape()};
            auto bn_node = std::shared_ptr<Node>(new op::BatchnormFprop(pattern_map[eps_label],
                                                                       pattern_map[gamma_label],
                                                                       pattern_map[beta_label],
                                                                       pattern_map[input],
                                                                       pattern_map[mean_label],
                                                                       pattern_map[variance_label],
                                                                       bn_output_shape));

            return bn_node;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(add_beta, callback);
        this->add_matcher(m);
    }

    TestGraphRewrite()
        : GraphRewrite()
    {
        construct_multiply_by_one();
        construct_add_zero();
        construct_sum();
	    construct_fprop_bn();
    }
};

static void run_passes(pass::Manager& pass_manager,
                       shared_ptr<Node> graph,
                       std::vector<shared_ptr<op::Parameter>> parms)
{
    auto func = make_shared<Function>(graph, op::Parameters{parms});
    pass_manager.run_passes(func);
}

TEST(pattern, graph_rewrite)
{
    auto shape = Shape{};
    pass::Manager pass_manager;
    pass_manager.register_pass<TestGraphRewrite>();

    {
        auto a = make_shared<op::Parameter>(element::i32, shape);
        auto b = make_shared<op::Parameter>(element::i32, shape);
        auto c = make_shared<op::Parameter>(element::i32, shape);
        auto iconst0 = construct_constant_node(0);
        auto graph_a = a + iconst0;
        auto graph_b = b + iconst0;

        auto f = std::make_shared<Function>(ngraph::Nodes{a, b, graph_a, c, graph_b},
                                            op::Parameters{a, b, c});
        pass_manager.run_passes(f);

        ASSERT_TRUE(graph_a->get_output_inputs(0).empty());
        ASSERT_TRUE(graph_b->get_output_inputs(0).empty());

        auto expected = ngraph::Nodes{a, b, a, c, b};
        ASSERT_TRUE(f->get_results() == expected);
    }

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

    //Sum rewrite
    {
        auto parm = make_shared<op::Parameter>(element::i32, Shape{2, 2});
        auto axes = AxisSet{0, 1};
        auto sum_graph = xla_sum(parm, axes);
        auto innermost_abs = make_shared<op::Abs>(sum_graph);

        auto nested_sum_graph = make_shared<op::Abs>(
            make_shared<op::Abs>(make_shared<op::Abs>(make_shared<op::Abs>(innermost_abs))));

        run_passes(pass_manager, nested_sum_graph, {parm});
        auto sum = std::dynamic_pointer_cast<op::Sum>(innermost_abs->get_input_op(0));
        ASSERT_TRUE(sum);
        ASSERT_EQ(sum->get_reduction_axes(), axes);
        ASSERT_EQ(sum->get_input_op(0), parm);
    }
}

TEST(pattern, matcher)
{
    auto shape = Shape{};
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
    auto label = std::make_shared<pattern::op::Label>(add, nullptr, Nodes{add});
    ASSERT_TRUE(n.match(label, add));
    ASSERT_EQ(n.get_pattern_map()[label], add);

    ASSERT_FALSE(n.match(label, a - b));

    ASSERT_TRUE(n.match(make_shared<op::Abs>(label), make_shared<op::Abs>(add)));
    ASSERT_EQ(n.get_pattern_map()[label], add);

    //Correlations
    auto label1 = std::make_shared<pattern::op::Label>(a);
    auto tmp = label1 + b;
    auto label2 = std::make_shared<pattern::op::Label>(tmp, nullptr, Nodes{tmp});
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
    auto variance  = std::make_shared<op::Divide>(xmu, N);

    auto var_graph = construct_variance_graph();
    ASSERT_TRUE(n.match(var_graph, variance));
    ASSERT_EQ(n.get_pattern_map()[var_graph], variance);    
}

TEST(pattern,  fuse_fprop_bn)
{
   pass::Manager pass_manager;
   pass_manager.register_pass<pass::VisualizeTree>("bn_fprop_before_fusion.pdf");
   pass_manager.register_pass<ngraph::pass::ReshapeElimination>();
   pass_manager.register_pass<pass::CPUFusion>();
   pass_manager.register_pass<pass::VisualizeTree>("bn_fprop_after_fusion.pdf");
   const string json_path = file_util::path_join(SERIALIZED_ZOO, "mxnet/bn_fprop_b2c3h2w2.json");
   const string json_string = file_util::read_file_to_string(json_path);
   stringstream ss(json_string);
   shared_ptr<Function> func = ngraph::deserialize(ss);
   pass_manager.run_passes(func);
}
