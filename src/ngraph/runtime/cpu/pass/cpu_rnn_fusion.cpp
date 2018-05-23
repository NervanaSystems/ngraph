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
#include <iostream>
#include <numeric>
#include <typeindex>
#include <typeinfo>
#include <unordered_set>

#include "cpu_rnn_fusion.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"
#include "ngraph/runtime/cpu/op/lstm.hpp"
#include "ngraph/runtime/cpu/op/rnn.hpp"
#include "ngraph/runtime/cpu/op/sigmoid.hpp"

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
                     << m.get_match_root()->get_name();
        auto pattern_map = m.get_pattern_map();

        if (m.get_match_root()->get_element_type() != element::f32)
        {
            NGRAPH_DEBUG << "mpattern = " << m.get_match_root()->get_name()
                         << " type is not float!";
            return false;
        }

        if (m.get_match_root()->get_outputs().size() != pattern_map[input]->get_outputs().size())
        {
            NGRAPH_DEBUG << "mpattern = " << m.get_match_root()->get_name()
                         << "input= " << pattern_map[input]->get_name() << "size dont match!";
            return false;
        }

        auto sigmoid_node = std::make_shared<op::Sigmoid>(pattern_map[input]);
        ngraph::replace_node(m.get_match_root(), sigmoid_node);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(divide_1_over_exp, callback);
    this->add_matcher(m);
}

void ngraph::runtime::cpu::pass::LSTMFusion::construct_lstm_fprop()
{
    auto input_xt = std::make_shared<pattern::op::Label>(element::f32, Shape{10, 100});
    auto weights_i2h = std::make_shared<pattern::op::Label>(element::f32, Shape{400, 100});
    auto weights_i2h_reshape =
        std::make_shared<op::Reshape>(weights_i2h, AxisVector{1, 0}, Shape{100, 400});
    auto dot_1 = std::make_shared<op::Dot>(input_xt, weights_i2h_reshape);

    auto bias_i2h = std::make_shared<pattern::op::Label>(element::f32, Shape{400});
    auto broadcast_bias_i2h = std::make_shared<op::Broadcast>(bias_i2h, Shape{10, 400}, AxisSet{0});
    auto add_1 = std::make_shared<op::Add>(dot_1, broadcast_bias_i2h);

    auto hidden_ht = std::make_shared<pattern::op::Label>(element::f32, Shape{10, 50});
    auto weights_h2h = std::make_shared<pattern::op::Label>(element::f32, Shape{400, 50});
    auto param2_2_reshape =
        std::make_shared<op::Reshape>(weights_h2h, AxisVector{1, 0}, Shape{50, 400});
    auto dot_2 = std::make_shared<op::Dot>(hidden_ht, param2_2_reshape);
    auto bias_h2h = std::make_shared<pattern::op::Label>(element::f32, Shape{400});
    auto broadcast_bias_h2h = std::make_shared<op::Broadcast>(bias_h2h, Shape{10, 400}, AxisSet{0});
    auto add_2 = std::make_shared<op::Add>(dot_2, broadcast_bias_h2h);

    auto X = std::make_shared<op::Add>(add_2, add_1);
    // construct forget gate
    auto input_slice_0 = std::make_shared<op::Slice>(X, Coordinate{0, 0}, Coordinate{10, 100});
    auto forget_gate = std::make_shared<op::Sigmoid>(input_slice_0);

    //ct-1 -> cell state (src_iter -> {ht | ct-1}
    auto ct_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{10, 100});
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
    pattern::graph_rewrite_callback callback = [ct_label,
                                                input_xt,
                                                weights_i2h,
                                                hidden_ht,
                                                weights_h2h,
                                                bias_i2h,
                                                bias_h2h,
                                                ct_1](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_fprop_lstm pattern against "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();
        NGRAPH_DEBUG << "In Lstm fprop call back";

        if (m.get_match_root()->get_element_type() != element::f32)
        {
            NGRAPH_DEBUG << "mpattern = " << m.get_match_root()->get_name()
                         << " type is not float!";
            return false;
        }

        auto input_xt_rank = input_xt->get_shape().size();
        auto hidden_ht_rank = hidden_ht->get_shape().size();
        auto weights_i2h_rank = weights_i2h->get_shape().size();
        auto weights_h2h_rank = weights_h2h->get_shape().size();
        if (input_xt_rank != 2 || hidden_ht_rank != 2 || weights_i2h_rank != 2 ||
            weights_h2h_rank != 2)
        {
            return false;
        }

        if (bias_i2h->get_shape().size() != 1 || bias_h2h->get_shape().size() != 1)
        {
            throw ngraph_error("Bias should have rank of 1 for MKLDNN Rnn op");
        }

        // Determine which is ht_1 and xt. but if both xt and ht_1 have the same shape we need to capture this
        // reliably in the RNN fusion.
        std::shared_ptr<op::Lstm> lstm = nullptr;
        bool intermediate_lstm = false;

        if (std::dynamic_pointer_cast<op::GetOutputElement>(pattern_map[ct_1]))
        {
            intermediate_lstm = true;
        }

        // this checks if its a first LSTM cell and uses constant initialization of hidden states to
        // differentiate between hidden state ht and input symbols xt.
        if (!intermediate_lstm &&
            (std::dynamic_pointer_cast<op::Broadcast>(pattern_map[hidden_ht]) &&
             std::dynamic_pointer_cast<op::Constant>(pattern_map[hidden_ht]->get_argument(0))))
        {
            lstm = std::make_shared<op::Lstm>(pattern_map[input_xt],
                                              pattern_map[weights_i2h],
                                              pattern_map[hidden_ht],
                                              pattern_map[weights_h2h],
                                              pattern_map[bias_i2h],
                                              pattern_map[bias_h2h],
                                              pattern_map[ct_1]);
        }
        else if (!intermediate_lstm &&
                 (std::dynamic_pointer_cast<op::Broadcast>(pattern_map[input_xt]) &&
                  std::dynamic_pointer_cast<op::Constant>(pattern_map[input_xt]->get_argument(0))))
        {
            lstm = std::make_shared<op::Lstm>(pattern_map[hidden_ht],
                                              pattern_map[weights_h2h],
                                              pattern_map[input_xt],
                                              pattern_map[weights_i2h],
                                              pattern_map[bias_h2h],
                                              pattern_map[bias_i2h],
                                              pattern_map[ct_1]);
        }
        else if (pattern_map[ct_1]->get_shape() == pattern_map[hidden_ht]->get_shape())
        {
            NGRAPH_DEBUG << "ct_shape : " << join(pattern_map[ct_1]->get_shape())
                         << " hidden state shape: " << join(pattern_map[hidden_ht]->get_shape());
            lstm = std::make_shared<op::Lstm>(pattern_map[input_xt],
                                              pattern_map[weights_i2h],
                                              pattern_map[hidden_ht],
                                              pattern_map[weights_h2h],
                                              pattern_map[bias_i2h],
                                              pattern_map[bias_h2h],
                                              pattern_map[ct_1]);
        }
        else
        {
            NGRAPH_DEBUG << "ct_shape: " << join(pattern_map[ct_1]->get_shape())
                         << " hidden state shape: " << join(pattern_map[input_xt]->get_shape());
            lstm = std::make_shared<op::Lstm>(pattern_map[hidden_ht],
                                              pattern_map[weights_h2h],
                                              pattern_map[input_xt],
                                              pattern_map[weights_i2h],
                                              pattern_map[bias_h2h],
                                              pattern_map[bias_i2h],
                                              pattern_map[ct_1]);
        }

        auto ht_output = std::make_shared<op::GetOutputElement>(lstm, 0);
        auto ct_output = std::make_shared<op::GetOutputElement>(lstm, 1);

        // Now identify the nodes which consumes the output of LSTM nodes
        // and replace them accordingly
        std::vector<std::shared_ptr<Node>> new_args;

        // find the user's for {ht|ct} and replace them with lstm_goe_1
        for (auto node : pattern_map[ct_label]->get_users())
        {
            NGRAPH_DEBUG << "node_name: " << node->get_name();
            for (size_t i = 0; i < node->get_input_size(); i++)
            {
                if (node->get_argument(i) == pattern_map[ct_label])
                {
                    node->get_inputs().at(i).replace_output(ct_output->get_outputs().at(0));
                }
            }
        }

        // find the user's for {ht} and replace them with lstm_goe_0
        ngraph::replace_node(m.get_match_root(), ht_output);
        return true;
    };
    auto m = std::make_shared<pattern::Matcher>(ht, callback);
    this->add_matcher(m);
}

static std::shared_ptr<ngraph::Node>
    compute_rnn_args(std::vector<std::shared_ptr<pattern::op::Label>>& rnn_labels,
                     pattern::RecurrentMatcher& m,
                     bool concat_all = false)
{
    NGRAPH_DEBUG << "Inside compute arg " << rnn_labels.size();
    NodeVector concat_args;

    // src_layer -> concatenate input symbols from different LSTM cells belonging to same RNN layer
    // in the order 0, 1, 2... t time slice
    if (concat_all)
    {
        auto node_labels = m.get_bound_nodes_for_pattern(rnn_labels[0]);
        std::reverse(node_labels.begin(), node_labels.end());
        return std::make_shared<op::Concat>(node_labels, 0);
    }

    // src_iter -> concatenate ht_1|ct_1 of the first LSTM cells belonging to same RNN layer
    if (rnn_labels.size() == 2)
    {
        for (size_t i = 0; i < rnn_labels.size(); i++)
        {
            auto node_labels = m.get_bound_nodes_for_pattern(rnn_labels[i]);
            // this is to make sure, we are not capturing any intermediate op's as Cell states.
            if (std::dynamic_pointer_cast<op::GetOutputElement>(
                    node_labels[node_labels.size() - 1]))
            {
                throw ngraph_error(
                    "pattern matcher error, ht_1|ct_1 of the first LSTM cell should not match "
                    "intermediate LSTM outputs");
            }
            concat_args.push_back(node_labels[node_labels.size() - 1]);
        }
        return std::make_shared<op::Concat>(concat_args, 0);
    }
    // i2h or h2h weights shared between LSTM cells
    else
    {
        auto node_labels = m.get_bound_nodes_for_pattern(rnn_labels[0]);
        return node_labels[node_labels.size() - 1];
    }
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

void ngraph::runtime::cpu::pass::RNNFusion::construct_rnn_lstm_fprop()
{
    auto ht_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{32, 100});
    auto weights_h2h = std::make_shared<pattern::op::Label>(element::f32, Shape{400, 100});
    auto xt = std::make_shared<pattern::op::Label>(element::f32, Shape{32, 100});
    auto weights_i2h = std::make_shared<pattern::op::Label>(element::f32, Shape{400, 100});
    auto bias_i2h = std::make_shared<pattern::op::Label>(element::f32, Shape{400});
    auto bias_h2h = std::make_shared<pattern::op::Label>(element::f32, Shape{400});
    auto rpattern_ct_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{32, 100});

    auto lstm = std::make_shared<op::Lstm>(
        xt, weights_i2h, ht_1, weights_h2h, bias_i2h, bias_h2h, rpattern_ct_1);
    auto goe = std::make_shared<op::GetOutputElement>(lstm, 0);
    auto lstm_node_label = std::make_shared<pattern::op::Label>(goe, nullptr, NodeVector{goe});

    pattern::recurrent_graph_rewrite_callback callback = [lstm_node_label,
                                                          xt,
                                                          weights_h2h,
                                                          ht_1,
                                                          weights_i2h,
                                                          bias_i2h,
                                                          bias_h2h,
                                                          rpattern_ct_1](
        pattern::RecurrentMatcher& m) {

        NGRAPH_DEBUG << " In recurrent RNN fusion callback";

        auto ht_1_label = m.get_bound_nodes_for_pattern(ht_1);

        // determine the ht and xt
        std::shared_ptr<ngraph::Node> src_layer = nullptr;
        std::shared_ptr<ngraph::Node> src_iter = nullptr;

        auto xt_node_array = m.get_bound_nodes_for_pattern(xt);
        auto hidden_ht_array = m.get_bound_nodes_for_pattern(ht_1);

        // since we dont have metadata to differentiate between xt and ht_1
        // we will be using the broadcasted constant initilization of the first LSTM cell
        // in the RNN layer to identify ht_1
        if (std::dynamic_pointer_cast<op::Broadcast>(xt_node_array[xt_node_array.size() - 1]) &&
            std::dynamic_pointer_cast<op::Constant>(
                xt_node_array[xt_node_array.size() - 1]->get_argument(0)))
        {
            std::vector<std::shared_ptr<pattern::op::Label>> src_layer_labels{ht_1};
            src_layer = compute_rnn_args(src_layer_labels, m, true);

            std::vector<std::shared_ptr<pattern::op::Label>> src_iter_labels{xt, rpattern_ct_1};
            src_iter = compute_rnn_args(src_iter_labels, m);
        }
        else if (std::dynamic_pointer_cast<op::Broadcast>(
                     hidden_ht_array[hidden_ht_array.size() - 1]) &&
                 std::dynamic_pointer_cast<op::Constant>(
                     hidden_ht_array[hidden_ht_array.size() - 1]->get_argument(0)))
        {
            std::vector<std::shared_ptr<pattern::op::Label>> src_layer_labels{xt};
            src_layer = compute_rnn_args(src_layer_labels, m, true);

            std::vector<std::shared_ptr<pattern::op::Label>> src_iter_labels{ht_1, rpattern_ct_1};
            src_iter = compute_rnn_args(src_iter_labels, m);
        }
        else
        {
            // dont fuse, if the PM didn't discover all the cells belonging to RNN layer.
            // we dont want to throw an assertion, if pattern matcher cannot discover all
            // nodes belonging to RNN, instead we will return and can compute LSTM cell wise
            return false;
        }

        std::vector<std::shared_ptr<pattern::op::Label>> weights_layer_labels{weights_i2h};
        auto weights_layer = compute_rnn_args(weights_layer_labels, m);

        std::vector<std::shared_ptr<pattern::op::Label>> weights_iter_labels{weights_h2h};
        auto weights_iter = compute_rnn_args(weights_iter_labels, m);
        auto bias_i2h_label = m.get_bound_nodes_for_pattern(bias_i2h);
        auto bias_h2h_label = m.get_bound_nodes_for_pattern(bias_h2h);
        auto bias = std::make_shared<op::Add>(bias_i2h_label[0], bias_h2h_label[0]);

        auto num_of_lstm_matched = m.get_number_of_recurrent_matches();
        size_t num_gates_in_lstm = 4;
        // TODO: assert for batch_size, sequence length and num_of_lstm's fused
        size_t batch_size = src_layer->get_shape()[0] / num_of_lstm_matched;
        size_t sequence_len = num_of_lstm_matched;
        size_t src_layer_feature_size = src_layer->get_shape()[1];
        size_t feature_size = ht_1_label[0]->get_shape()[1];
        // number of states for LSTM is 2
        size_t num_cell_states = 2;
        size_t direction = 1;
        size_t num_fused_rnn_layers = 1;

        NGRAPH_DEBUG << "src_layer: " << join(src_layer->get_shape());
        NGRAPH_DEBUG << "src_iter: " << join(src_iter->get_shape());
        NGRAPH_DEBUG << "weights_layer: " << join(weights_layer->get_shape());
        NGRAPH_DEBUG << "weights_iter: " << join(weights_iter->get_shape());
        NGRAPH_DEBUG << "bias: " << join(bias->get_shape());
        NGRAPH_DEBUG << "src_seq_len: " << sequence_len;
        NGRAPH_DEBUG << "batch_size: " << batch_size;
        NGRAPH_DEBUG << "feature_size: " << feature_size;

        if ((src_layer->get_arguments().size()) != sequence_len)
        {
            throw ngraph_error(
                "number of lstm inputs captured in the RNN fusion is not equal to "
                "src_sequence_length");
        }

        if ((src_iter->get_arguments().size()) != num_cell_states)
        {
            throw ngraph_error("number of states for RNN op is not equal to (ht_1|ct_1)");
        }

        auto src_layer_rank = src_layer->get_shape().size();
        auto src_iter_rank = src_iter->get_shape().size();
        auto weights_layer_rank = weights_layer->get_shape().size();
        auto weights_iter_rank = weights_iter->get_shape().size();
        auto bias_rank = bias->get_shape().size();
        if (src_layer_rank != 2 || src_iter_rank != 2 || weights_layer_rank != 2 ||
            weights_iter_rank != 2)
        {
            throw ngraph_error(
                "Pattern matcher error src_layer, weights_layer, src_iter, weights_iter should "
                "have rank 2 for MKLDNN RNN op");
        }

        if (bias_rank != 1)
        {
            throw ngraph_error("Bias should have rank of 1 for MKLDNN Rnn op");
        }

        if (src_layer->get_element_type() != element::f32 ||
            src_iter->get_element_type() != element::f32)
        {
            throw ngraph_error(
                "input tensor type and input recurrent state tensor type for MKLDNN RNN op should "
                "be float32");
        }

        auto rnn = std::make_shared<op::Rnn>(src_layer,
                                             src_iter,
                                             weights_layer,
                                             weights_iter,
                                             bias,
                                             num_of_lstm_matched,
                                             num_gates_in_lstm,
                                             sequence_len,
                                             src_layer_feature_size,
                                             feature_size,
                                             num_cell_states,
                                             direction,
                                             num_fused_rnn_layers);

        std::vector<std::shared_ptr<op::Slice>> ht_slice_per_timestep(num_of_lstm_matched, nullptr);
        auto rnn_ht_out = std::make_shared<op::GetOutputElement>(rnn, 0);
        auto rnn_ct_out = std::make_shared<op::GetOutputElement>(rnn, 1);

        //slice the rnn ht's
        size_t start_index = 0;
        size_t end_index = batch_size;
        // capture the slices in the reverse order, so it corrosponds to lstm_goes order captured by the Pattern matcher
        for (size_t i = 0; i < num_of_lstm_matched; i++)
        {
            ht_slice_per_timestep[i] = (std::make_shared<op::Slice>(
                rnn_ht_out, Coordinate{start_index, 0}, Coordinate{end_index, feature_size}));
            start_index += batch_size;
            end_index += batch_size;
        }
        std::reverse(ht_slice_per_timestep.begin(), ht_slice_per_timestep.end());

        NGRAPH_DEBUG << "rnn_time_slice: " << ht_slice_per_timestep.size();

        // find the lstm's nodes captured in PM
        auto lstm_goes = m.get_bound_nodes_for_pattern(lstm_node_label);
        std::vector<std::shared_ptr<ngraph::Node>> lstm_nodes;

        // we need to collect LSTM from GOE's, in order to deterministicaly determine
        // the individaual time slice output ht. lstm_goes will hold the GOE in the decreasing
        // order of the time slices
        for (size_t i = 0; i < lstm_goes.size(); i++)
        {
            // lstm's will be the input to GOE's
            lstm_nodes.push_back(lstm_goes[i]->get_arguments()[0]);
        }

        if (sequence_len != lstm_nodes.size())
        {
            throw ngraph_error(" Number of lstm nodes in RNN layer is not equal to time slices");
        }

        if (lstm_nodes.size() != lstm_goes.size() &&
            lstm_goes.size() != ht_slice_per_timestep.size())
        {
            throw ngraph_error(
                "Number of slices of rnn output ht is not equal to the time slices in RNN layer");
        }

        // collect all the consumers of LSTM goe's (ht)
        std::set<std::shared_ptr<ngraph::Node>> lstm_goe0_user;
        std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node>> map_goe_to_lstm_slices;
        std::shared_ptr<Node> goe_0;
        for (size_t index = 0; index < lstm_nodes.size(); index++)
        {
            // now get the GOE0 which is the first output of lstm (ht)
            for (auto& goes : lstm_nodes[index]->get_outputs().at(0).get_inputs())
            {
                auto goe_node = std::dynamic_pointer_cast<op::GetOutputElement>(goes->get_node());
                // first output node of lstm
                if (goe_node->get_n() == 0)
                {
                    goe_0 = goes->get_node();
                }
            }

            for (auto goe0_user : goe_0->get_users())
            {
                if (std::find(lstm_nodes.begin(), lstm_nodes.end(), goe0_user) ==
                        lstm_nodes.end() &&
                    !is_unreachable(goe0_user))
                {
                    lstm_goe0_user.insert(goe0_user);
                    map_goe_to_lstm_slices[goe_0] = ht_slice_per_timestep[index];
                    NGRAPH_DEBUG << "ht_slice: " << ht_slice_per_timestep[index]->get_name()
                                 << " goe0_user " << goe0_user->get_name() << " ";
                }
            }
        }

        //now go through the lstm consumers and replace them with the slice
        for (auto& node : lstm_goe0_user)
        {
            for (size_t i = 0; i < node->get_input_size(); i++)
            {
                if (map_goe_to_lstm_slices.find(node->get_argument(i)) !=
                    map_goe_to_lstm_slices.end())
                {
                    node->get_inputs().at(i).replace_output(
                        map_goe_to_lstm_slices[node->get_argument(i)]->get_outputs().at(0));
                }
            }
        }
        NGRAPH_DEBUG << "End of recurrent fusion call back "
                     << "matched_node: " << m.get_match_root()->get_name();
        return true;

    };

    std::set<std::shared_ptr<pattern::op::Label>> empty_correlated_matches;
    auto m = std::make_shared<pattern::RecurrentMatcher>(
        lstm_node_label, rpattern_ct_1, empty_correlated_matches, callback);
    this->add_matcher(m);
}
