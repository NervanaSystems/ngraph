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
#include <typeindex>
#include <typeinfo>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <unordered_set>
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/any.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/runtime/cpu/op/batch_norm_relu.hpp"
#include "ngraph/runtime/cpu/op/conv_bias.hpp"
#include "ngraph/runtime/cpu/op/conv_relu.hpp"
#include "ngraph/runtime/cpu/op/lstm.hpp"
#include "ngraph/runtime/cpu/op/matmul_bias.hpp"
#include "ngraph/runtime/cpu/op/rnn.hpp"
#include "ngraph/runtime/cpu/op/sigmoid.hpp"
#include "rnn_fusion.hpp"

using namespace ngraph;
void ngraph::runtime::cpu::pass::LSTMFusion::construct_sigmoid()
{
    //construct variance
    auto input = std::make_shared<pattern::op::Label>(element::f32, Shape{3, 4});
    auto neg_input = std::make_shared<op::Negative>(input);
    auto exp_neg_input = std::make_shared<op::Exp>(neg_input);

    // broadcast input
    auto constant = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto broadcast_constant = std::make_shared<op::Broadcast>(constant, Shape{3, 4}, AxisSet{0, 1});

    auto add_exp = std::make_shared<op::Add>(exp_neg_input, broadcast_constant);
    auto divide_1_over_exp = std::make_shared<op::Divide>(broadcast_constant, add_exp);

    //Define a call back that needs to called once the DFG matches the pattern
    ngraph::pattern::graph_rewrite_callback callback = [input](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_fprop_sigmoid pattern against "
                     << m.match_root()->get_name();
        auto pattern_map = m.get_pattern_map();

        if (m.match_root()->get_element_type() != element::f32)
        {
            NGRAPH_DEBUG << "mpattern = " << m.match_root()->get_name() << " type is not float!";
            return false;
        }

        if (m.match_root()->get_outputs().size() != pattern_map[input]->get_outputs().size())
        {
            NGRAPH_DEBUG << "mpattern = " << m.match_root()->get_name()
                         << "input= " << pattern_map[input]->get_name() << "size dont match!";
            return false;
        }

        auto sigmoid_node = std::make_shared<op::Sigmoid>(pattern_map[input]);
        ngraph::replace_node(m.match_root(), sigmoid_node);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(divide_1_over_exp, callback);
    std::cout << "Sigmoid: " << m << std::endl;
    this->add_matcher(m);
}

void ngraph::runtime::cpu::pass::LSTMFusion::construct_lstm_fprop()
{
    // param1_1 -> ht_1 (src_iter)
    auto param1_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{10, 100});
    auto broadcast_pred_1 = [](std::shared_ptr<Node> n) {
        return static_cast<bool>(std::dynamic_pointer_cast<op::Broadcast>(n));
    };
    auto skip_param_1_1 = std::make_shared<pattern::op::Any>(param1_1, broadcast_pred_1);
    // param1_2 -> h2h weights (weights_iter)
    auto param1_2 = std::make_shared<pattern::op::Label>(element::f32, Shape{400, 100});
    auto param1_2_reshape =
        std::make_shared<op::Reshape>(param1_2, AxisVector{1, 0}, Shape{100, 400});
    auto dot_1 = std::make_shared<op::Dot>(skip_param_1_1, param1_2_reshape);

    auto bias1 = std::make_shared<pattern::op::Label>(element::f32, Shape{400});
    auto broadcast_bias1 = std::make_shared<op::Broadcast>(bias1, Shape{10, 400}, AxisSet{0});
    auto add_1 = std::make_shared<op::Add>(dot_1, broadcast_bias1);

    // param2_1 -> xt (src_layer)
    auto param2_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{10, 50});
    // param2_2 -> i2h weights (weights_layer)
    auto param2_2 = std::make_shared<pattern::op::Label>(element::f32, Shape{400, 50});
    auto param2_2_reshape =
        std::make_shared<op::Reshape>(param2_2, AxisVector{1, 0}, Shape{50, 400});
    auto dot_2 = std::make_shared<op::Dot>(param2_1, param2_2_reshape);
    auto bias2 = std::make_shared<pattern::op::Label>(element::f32, Shape{400});
    auto broadcast_bias2 = std::make_shared<op::Broadcast>(bias2, Shape{10, 400}, AxisSet{0});
    auto add_2 = std::make_shared<op::Add>(dot_2, broadcast_bias2);

    auto X = std::make_shared<op::Add>(add_2, add_1);
    // construct forget gate
    auto input_slice_0 = std::make_shared<op::Slice>(X, Coordinate{0, 0}, Coordinate{10, 100});
    auto forget_gate = std::make_shared<op::Sigmoid>(input_slice_0);

    //ct-1 -> cell state (src_iter -> {ht | ct-1}
    auto ct_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{10, 100});
    //auto skip_ct_1 = std::make_shared<pattern::op::Any>(ct_1, broadcast_pred);
    auto multiply_forget_gate_ct_1 = std::make_shared<op::Multiply>(forget_gate, ct_1);

    // construct input gate
    auto input_slice_1 = std::make_shared<op::Slice>(X, Coordinate{0, 100}, Coordinate{10, 200});
    auto input_gate = std::make_shared<op::Sigmoid>(input_slice_1);
    auto input_slice_2 = std::make_shared<op::Slice>(X, Coordinate{0, 200}, Coordinate{10, 300});
    auto tanh_1 = std::make_shared<op::Tanh>(input_slice_2);
    auto multiply_input_gate_tanh_1 = std::make_shared<op::Multiply>(input_gate, tanh_1);

    auto add_ct_1_input_gate_tanh_1 =
        std::make_shared<op::Add>(multiply_forget_gate_ct_1, multiply_input_gate_tanh_1);
    auto ct_label = std::make_shared<pattern::op::Label>(
        add_ct_1_input_gate_tanh_1, nullptr, NodeVector{add_ct_1_input_gate_tanh_1});

    // construct output gate
    auto input_slice_3 = std::make_shared<op::Slice>(X, Coordinate{0, 300}, Coordinate{10, 400});
    auto output_gate = std::make_shared<op::Sigmoid>(input_slice_3);
    auto tanh_2 = std::make_shared<op::Tanh>(ct_label);
    auto ht = std::make_shared<op::Multiply>(output_gate, tanh_2);
    auto ht_label = std::make_shared<pattern::op::Label>(ht, nullptr, NodeVector{ht});

    //Define a call back that needs to called once the DFG matches the pattern
    pattern::graph_rewrite_callback callback =
        [ct_label, param1_1, param1_2, param2_1, param2_2, bias1, bias2, ct_1](
            pattern::Matcher& m) {
            NGRAPH_DEBUG << "In a callback for construct_fprop_lstm pattern against "
                         << m.match_root()->get_name();

            auto pattern_map = m.get_pattern_map();
            std::cout << "In Lstm fprop call back" << std::endl;

            // if (m.match_root()->get_element_type() != element::f32)
            // {
            //     NGRAPH_DEBUG << "mpattern = " << m.match_root()->get_name() << " type is not float!";
            //     return false;
            // }

            // if (m.match_root()->get_outputs().size() != pattern_map[input]->get_outputs().size())
            // {
            //     NGRAPH_DEBUG << "mpattern = " << m.match_root()->get_name()
            //                  << "input= " << pattern_map[input]->get_name() << "size dont match!";
            //     return false;
            // }

            //std::cout << "label_ct: " << join(label_ct[0]->get_shape()) <<  " " << label_ct[0]->get_name() << std::endl;
            Shape ct_shape{pattern_map[ct_label]->get_shape()};
            auto lstm = std::make_shared<op::Lstm>(pattern_map[param1_1],
                                                   pattern_map[param1_2],
                                                   pattern_map[param2_1],
                                                   pattern_map[param2_2],
                                                   pattern_map[bias1],
                                                   pattern_map[bias2],
                                                   pattern_map[ct_1],
                                                   ct_shape);

            auto ht_output = std::make_shared<op::GetOutputElement>(lstm, 0);
            auto ct_output = std::make_shared<op::GetOutputElement>(lstm, 1);

            std::vector<std::shared_ptr<Node>> new_args;
            for (auto node : pattern_map[ct_label]->get_users())
            {
                //std::cout << "Add_inputs: " << node->get_name() << std::endl;
                if (std::dynamic_pointer_cast<op::Multiply>(node))
                {
                    std::cout << "node_name: " << node->get_name() << std::endl;
                    for (size_t i = 0; i < node->get_input_size(); i++)
                    {
                        if (node->get_argument(i) == pattern_map[ct_label])
                        {
                            new_args.push_back(ct_output);
                        }
                        else
                        {
                            new_args.push_back(node->get_argument(i));
                        }
                        std::cout << "Multiply_input's shape: " << join(new_args[i]->get_shape())
                                  << " " << new_args[i]->get_name() << std::endl;
                    }
                    auto new_ct_node = node->copy_with_new_args(new_args);
                    std::cout << "node: " << node->get_name() << " replaced with  "
                              << new_ct_node->get_name() << std::endl;
                    ;
                    ngraph::replace_node(node, new_ct_node);
                    new_args.clear();
                }
            }
            ngraph::replace_node(m.match_root(), ht_output);
            return true;
        };
    //std::set<std::shared_ptr<pattern::op::Label>> empty_correlated_matches;
    auto m = std::make_shared<pattern::Matcher>(ht, callback);
    std::cout << "lstm: " << m << std::endl;
    this->add_matcher(m);
}

static std::shared_ptr<ngraph::Node>
    compute_rnn_args(std::vector<std::shared_ptr<pattern::op::Label>>& rnn_labels,
                     pattern::RecurrentMatcher& m,
                     bool input_symbol = false)
{
    std::cout << "Inside compute arg " << rnn_labels.size() << std::endl;
    std::set<std::shared_ptr<Node>> unique_params;
    NodeVector concat_args;
    for (size_t i = 0; i < rnn_labels.size(); i++)
    {
        auto node_lables = m.get_bound_nodes_for_pattern(rnn_labels[i]);
        // std::cout << "rnn_label: " << node_lables[0]->get_name() << " "
        //           << join(node_lables[0]->get_shape()) << " ";
        for (size_t j = 0; j < node_lables.size(); j++)
        {
            if (!std::dynamic_pointer_cast<op::GetOutputElement>(node_lables[j]) && !input_symbol)
            {
                unique_params.insert(node_lables[j]);
            }
            if (input_symbol)
            {
                unique_params.insert(node_lables[j]);
            }
        }
    }
    // push the uniques params as the Rnn arguments
    if (!unique_params.empty())
    {
        for (auto& param : unique_params)
        {
            //concat all the bounded params
            std::cout << "concat_args: " << param->get_name() << " " << join(param->get_shape())
                      << std::endl;
            concat_args.push_back(param);
        }
        if (concat_args.size() > 1)
        {
            // reverse the concat_args so we concat in order of 0th, 1st,2nd....t'th time slice
            std::reverse(concat_args.begin(), concat_args.end());
            return std::make_shared<op::Concat>(concat_args, 0);
        }
    }
    return concat_args[0];
}

static bool is_unreachable(std::shared_ptr<ngraph::Node> node)
{
    std::unordered_set<std::shared_ptr<ngraph::Node>> instances_seen;
    std::deque<std::shared_ptr<ngraph::Node>> stack;
    stack.push_front(node);

    while (stack.size() > 0)
    {
        std::shared_ptr<ngraph::Node> n = stack.front();
        if (instances_seen.count(n) == 0)
        {
            if (n->is_output())
            {
                return false;
            }
            instances_seen.insert(n);
        }
        stack.pop_front();
        for (auto arg : n->get_users())
        {
            if (instances_seen.count(arg) == 0)
            {
                stack.push_front(arg);
            }
        }
    }
    return true;
}

void ngraph::runtime::cpu::pass::RNNFusion::construct_rnn_fprop()
{
    auto rpattern_ht_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{32, 100});
    //auto skip_ht_1 = std::make_shared<pattern::op::Any>(rpattern_ht_1, broadcast_pred);
    auto weights_h2h = std::make_shared<pattern::op::Label>(element::f32, Shape{400, 100});
    auto xt = std::make_shared<pattern::op::Label>(element::f32, Shape{32, 200});
    auto weights_i2h = std::make_shared<pattern::op::Label>(element::f32, Shape{400, 100});
    auto bias1 = std::make_shared<pattern::op::Label>(element::f32, Shape{400});
    auto bias2 = std::make_shared<pattern::op::Label>(element::f32, Shape{400});
    auto ct_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{32, 100});

    auto lstm = std::make_shared<op::Lstm>(
        xt, weights_i2h, rpattern_ht_1, weights_h2h, bias1, bias2, ct_1, Shape{32, 100});
    auto goe = std::make_shared<op::GetOutputElement>(lstm, 0);
    auto lstm_node_label = std::make_shared<pattern::op::Label>(goe, nullptr, NodeVector{goe});

    pattern::recurrent_graph_rewrite_callback callback =
        [lstm_node_label, rpattern_ht_1, weights_h2h, xt, weights_i2h, bias1, bias2, ct_1](
            pattern::RecurrentMatcher& m) {

            // static int count = 0;
            // if (count++ > 0)
            // return false;
            std::cout << "|||||||| In recurrent fusion |||||||" << std::endl;

            auto ht_1_label = m.get_bound_nodes_for_pattern(rpattern_ht_1);

            std::vector<std::shared_ptr<pattern::op::Label>> src_iter_labels{rpattern_ht_1, ct_1};
            auto src_iter = compute_rnn_args(src_iter_labels, m);

            std::vector<std::shared_ptr<pattern::op::Label>> weights_layer_labels{weights_i2h};
            auto weights_layer = compute_rnn_args(weights_layer_labels, m);

            std::vector<std::shared_ptr<pattern::op::Label>> weights_iter_labels{weights_h2h};
            auto weights_iter = compute_rnn_args(weights_iter_labels, m);

            std::vector<std::shared_ptr<pattern::op::Label>> src_layer_labels{xt};
            auto src_layer = compute_rnn_args(src_layer_labels, m, true);

            auto bias_i2h_label = m.get_bound_nodes_for_pattern(bias2);
            auto bias_h2h_label = m.get_bound_nodes_for_pattern(bias1);
            auto bias = std::make_shared<op::Add>(bias_i2h_label[0], bias_h2h_label[0]);

            auto num_of_lstm_matched = m.get_number_of_recurrent_matches();
            size_t num_gates_in_lstm = 4;
            // TODO: assert for batch_size, sequence length and num_of_lstm's fused
            size_t batch_size = src_layer->get_shape()[0] / num_of_lstm_matched;
            size_t sequence_len = num_of_lstm_matched;
            size_t feature_size = ht_1_label[0]->get_shape()[1];
            // number of states for LSTM is 2
            size_t num_rnn_cell_states = 2;

            auto rnn = std::make_shared<op::Rnn>(src_layer,
                                                 src_iter,
                                                 weights_layer,
                                                 weights_iter,
                                                 bias,
                                                 num_of_lstm_matched,
                                                 num_gates_in_lstm,
                                                 sequence_len,
                                                 feature_size,
                                                 num_rnn_cell_states);

            std::cout << "src_layer: " << join(src_layer->get_shape()) << std::endl;
            std::cout << "src_iter: " << join(src_iter->get_shape()) << std::endl;
            std::cout << "weights_layer: " << join(weights_layer->get_shape()) << std::endl;
            std::cout << "weights_iter: " << join(weights_iter->get_shape()) << std::endl;
            std::cout << "bias: " << join(bias->get_shape()) << std::endl;

            std::vector<std::shared_ptr<op::Slice>> ht_slice_per_timestep;
            auto rnn_ht_out = std::make_shared<op::GetOutputElement>(rnn, 0);
            auto rnn_ct_out = std::make_shared<op::GetOutputElement>(rnn, 1);

            //slice the rnn ht's
            size_t start_index = 0;
            size_t end_index = batch_size;
            for (size_t i = 0; i < num_of_lstm_matched; i++)
            {
                ht_slice_per_timestep.push_back(std::make_shared<op::Slice>(
                    rnn_ht_out, Coordinate{start_index, 0}, Coordinate{end_index, feature_size}));
                start_index += batch_size;
                end_index += batch_size;
            }

            std::cout << "rnn_time_slice: " << ht_slice_per_timestep.size() << std::endl;

            // find the lstm's nodes captured in PM
            auto lstm_goes = m.get_bound_nodes_for_pattern(lstm_node_label);
            std::set<std::shared_ptr<ngraph::Node>> lstm_nodes;
            for (size_t i = 0; i < lstm_goes.size(); i++)
            {
                // lstm's will be the input to GOE's
                lstm_nodes.insert(lstm_goes[i]->get_arguments()[0]);
            }

            // collect all the consumers of LSTM goe's (ht)
            std::set<std::shared_ptr<ngraph::Node>> lstm_goe0_user;
            std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node>> map_nodes_to_goe;
            std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node>> map_goe_to_lstm_slices;
            std::shared_ptr<Node> goe_0;
            //TODO figure out to elimate using buggy index
            size_t index = 0;
            for (auto& node : lstm_nodes)
            {
                // now get the GOE0 which is the first output of lstm (ht)
                for (auto& goes : node->get_outputs().at(0).get_inputs())
                {
                    auto goe_node =
                        std::dynamic_pointer_cast<op::GetOutputElement>(goes->get_node());
                    // first output node of lstm
                    if (goe_node->get_n() == 0)
                    {
                        goe_0 = goes->get_node();
                    }
                }

                for (auto goe0_user : goe_0->get_users())
                {
                    if (lstm_nodes.find(goe0_user) == lstm_nodes.end() &&
                        !is_unreachable(goe0_user))
                    {
                        lstm_goe0_user.insert(goe0_user);
                        map_goe_to_lstm_slices[goe0_user] = ht_slice_per_timestep[index];
                        std::cout << "goe0_user " << goe0_user->get_name() << " ";
                    }
                }
                index++;
            }

            std::cout << "++ done collecting all lstm users ++++ " << std::endl;
            //now go through the lstm consumers and replace them with the slice
            std::vector<std::shared_ptr<Node>> new_args;
            for (auto& node : lstm_goe0_user)
            {
                for (auto& node_args : node->get_arguments())
                {
                    if (std::find(lstm_goes.begin(), lstm_goes.end(), node_args) != lstm_goes.end())
                    {
                        std::cout << "index: " << index << " args_shape "
                                  << join(node_args->get_shape())
                                  << "name: " << node_args->get_name() << std::endl;
                        new_args.push_back(map_goe_to_lstm_slices[node]);
                    }
                    else
                    {
                        std::cout << " args_shape " << join(node_args->get_shape())
                                  << "name: " << node_args->get_name() << std::endl;
                        new_args.push_back(node_args);
                    }
                }
                std::cout << "node bring replaced " << node->get_name();
                auto new_node = node->copy_with_new_args(new_args);
                ngraph::replace_node(node, new_node);
                std::cout << "node: " << node->get_name() << " replaced with  "
                          << new_node->get_name() << std::endl;
                new_args.clear();
            }
            std::cout << "<<<<<<<<<<<< End recurrent fusion >>>>>>>>>>>>>" << std::endl;
            ngraph::replace_node(m.get_match_root(),
                                 ht_slice_per_timestep[ht_slice_per_timestep.size() - 1]);
            return true;

        };

    std::set<std::shared_ptr<pattern::op::Label>> empty_correlated_matches;
    auto m = std::make_shared<pattern::RecurrentMatcher>(
        lstm_node_label, rpattern_ht_1, empty_correlated_matches, callback);
    this->add_matcher(m);
}
