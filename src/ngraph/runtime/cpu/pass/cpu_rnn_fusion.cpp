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

#include <algorithm>
#include <iostream>
#include <numeric>
#include <typeindex>
#include <typeinfo>
#include <unordered_set>

#include "cpu_rnn_fusion.hpp"
#include "ngraph/descriptor/output.hpp"
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
#include "ngraph/op/reverse_sequence.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"
#include "ngraph/runtime/cpu/op/lstm.hpp"
#include "ngraph/runtime/cpu/op/rnn.hpp"
#include "ngraph/runtime/cpu/op/sigmoid.hpp"

#define STR(X) #X
#define CHECK_RANK(X, RANK)                                                                        \
    if (pattern_map[X]->get_shape().size() != RANK)                                                \
    {                                                                                              \
        NGRAPH_DEBUG << STR(X) << " does not have rank " << RANK;                                  \
        return false;                                                                              \
    }

using namespace ngraph;
void ngraph::runtime::cpu::pass::LSTMFusion::construct_sigmoid()
{
    // construct variance
    auto input = std::make_shared<pattern::op::Label>(element::f32, Shape{3, 4});
    auto neg_input = std::make_shared<op::Negative>(input);
    auto exp_neg_input = std::make_shared<op::Exp>(neg_input);

    // broadcast input
    auto constant = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto broadcast_constant = std::make_shared<op::Broadcast>(constant, Shape{3, 4}, AxisSet{0, 1});

    auto add_exp = std::make_shared<op::Add>(exp_neg_input, broadcast_constant);
    auto divide_1_over_exp = std::make_shared<op::Divide>(broadcast_constant, add_exp);

    // Define a call back that needs to called once the DFG matches the pattern
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

static void replace_collapse_node_user(std::shared_ptr<Node> collapsed_node,
                                       descriptor::Output& new_output)
{
    for (auto node : collapsed_node->get_users(true))
    {
        NGRAPH_DEBUG << "node_name: " << node->get_name();
        for (size_t i = 0; i < node->get_input_size(); i++)
        {
            if (node->get_argument(i) == collapsed_node)
            {
                node->get_inputs().at(i).replace_output(new_output);
            }
        }
    }
}

void ngraph::runtime::cpu::pass::LSTMFusion::construct_lstm_fprop()
{
    // This pattern captures the following equations in the given data
    // flow graph
    //
    //   i_t = sigmoid(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi});
    //   f_t = sigmoid(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf});
    //   g_t = tanh   (W_{ig} x_t + b_{ig} + W_{hg} h_{(t-1)} + b_{hg});
    //   o_t = sigmoid(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho});
    //   c_t = f_t * c_{(t-1)} + i_t * g_t;
    //   h_t = o_t * \ tanh(c_t);
    //

    // Inputs to the sub-graph
    // Assumes weights for all the 4 gates are fused in the order -
    //                      input (i), forget (f), block (g) and output (o)
    auto w_i2h = std::make_shared<pattern::op::Label>(element::f32, Shape{100, 400});
    auto bias_i2h = std::make_shared<pattern::op::Label>(element::f32, Shape{10, 400});
    auto w_h2h = std::make_shared<pattern::op::Label>(element::f32, Shape{50, 400});
    auto bias_h2h = std::make_shared<pattern::op::Label>(element::f32, Shape{10, 400});
    auto xt = std::make_shared<pattern::op::Label>(element::f32, Shape{10, 100});
    auto ht_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{10, 50});
    auto ct_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{10, 100});

    auto broadcast_pred = [](std::shared_ptr<Node> n) {
        return ((std::dynamic_pointer_cast<op::Broadcast>(n) != nullptr) ||
                (std::dynamic_pointer_cast<op::Reshape>(n) != nullptr));
    };

    // Fused MatMuls
    // (W_{ii} | (W_{if} | W_{ig} | W_{io}) * x_t + (b_{ii} | b_{if} |  b_{ig} | b_{io})
    auto dot1 = std::make_shared<op::Dot>(xt, w_i2h);
    auto add1 = std::make_shared<op::Add>(
        dot1, std::make_shared<pattern::op::Skip>(bias_i2h, broadcast_pred));
    // (W_{hi} | (W_{hf} | W_{hg} | W_{ho}) * h_{(t-1)} + (b_{hi} | b_{hf} |  b_{hg} | b_{ho})
    auto dot2 = std::make_shared<op::Dot>(ht_1, w_h2h);
    auto add2 = std::make_shared<op::Add>(
        dot2, std::make_shared<pattern::op::Skip>(bias_h2h, broadcast_pred));

    auto X = std::make_shared<op::Add>(add2, add1);

    // construct gates
    auto it = std::make_shared<op::Sigmoid>(
        std::make_shared<op::Slice>(X, Coordinate{0, 0}, Coordinate{10, 100}));
    auto ft = std::make_shared<op::Sigmoid>(
        std::make_shared<op::Slice>(X, Coordinate{0, 100}, Coordinate{10, 200}));
    auto gt = std::make_shared<op::Tanh>(
        std::make_shared<op::Slice>(X, Coordinate{0, 200}, Coordinate{10, 300}));
    auto ot = std::make_shared<op::Sigmoid>(
        std::make_shared<op::Slice>(X, Coordinate{0, 300}, Coordinate{10, 400}));

    // construct (c_t) cell state
    auto ct = std::make_shared<op::Add>(std::make_shared<op::Multiply>(ft, ct_1),
                                        std::make_shared<op::Multiply>(it, gt));
    auto ct_label = std::make_shared<pattern::op::Label>(ct, nullptr, NodeVector{ct});

    // construct (h_t)
    auto ht = std::make_shared<op::Multiply>(ot, std::make_shared<op::Tanh>(ct_label));

    // Define a call back that needs to called once the DFG matches the pattern
    pattern::graph_rewrite_callback callback =
        [ct_label, w_i2h, bias_i2h, w_h2h, bias_h2h, xt, ht_1, ct_1](pattern::Matcher& m) {
            NGRAPH_DEBUG << "In a callback for construct_fprop_lstm pattern against "
                         << m.get_match_root()->get_name();

            auto pattern_map = m.get_pattern_map();

            if (m.get_match_root()->get_element_type() != element::f32)
            {
                NGRAPH_DEBUG << "mpattern = " << m.get_match_root()->get_name()
                             << " type is not float!";
                return false;
            }

            CHECK_RANK(xt, 2);
            CHECK_RANK(ht_1, 2);
            CHECK_RANK(w_i2h, 2);
            CHECK_RANK(w_h2h, 2);
            CHECK_RANK(bias_i2h, 1);
            CHECK_RANK(bias_h2h, 1);

            auto weights_layer = pattern_map[w_i2h];
            auto weights_iter = pattern_map[w_h2h];
            auto src_layer = pattern_map[xt];
            auto hidden_state = pattern_map[ht_1];
            auto cell_state = pattern_map[ct_1];

            // TODO: (Pruthvi) temporary workaround for GNMT slow down
            // this checks avoids fusing of LSTM cells if its a part of decoder, we
            // will remove this once mkldnn optimizes individual LSTM cell or once
            // we have decoder pattern for GNMT.
            if (!(std::dynamic_pointer_cast<op::Broadcast>(cell_state) &&
                  std::dynamic_pointer_cast<op::Constant>(cell_state->get_argument(0))) &&
                !(std::dynamic_pointer_cast<op::Slice>(cell_state) &&
                  std::dynamic_pointer_cast<op::GetOutputElement>(cell_state->get_argument(0))))
            {
                return false;
            }

            auto swap_lstm_inputs = [&]() -> void {
                src_layer = pattern_map[ht_1];
                hidden_state = pattern_map[xt];
                weights_layer = pattern_map[w_h2h];
                weights_iter = pattern_map[w_i2h];
            };

            // LSTM kernel expects ht_1 and ct_1 to have the same shape but the
            // pattern matcher cannot guarantee this since the computations are
            // symmetric around x_t and ht_1. Use heuristics to swap the matched
            // labels
            if (std::dynamic_pointer_cast<op::Broadcast>(src_layer) &&
                std::dynamic_pointer_cast<op::Constant>(src_layer->get_argument(0)))
            {
                // First timestep of an RNN layer
                swap_lstm_inputs();
            }
            else if (hidden_state->get_shape() != cell_state->get_shape())
            {
                swap_lstm_inputs();
            }
            else if (std::dynamic_pointer_cast<op::GetOutputElement>(cell_state->get_argument(0)))
            {
                // swap the inputs if the cell_state and hidden state does not
                // belong to the same Lstm
                if (hidden_state->get_argument(0)->get_arguments()[0] !=
                    cell_state->get_argument(0)->get_arguments()[0])
                {
                    swap_lstm_inputs();
                }
            }

            if (hidden_state->get_shape() != cell_state->get_shape())
            {
                NGRAPH_DEBUG
                    << "Lstm MKLDNN kernel requires recurrent output hidden states to match ";
                return false;
            }

            // set LSTM cell attributes
            size_t lstm_n_gates = 4;
            size_t batch_size = src_layer->get_shape()[0];
            size_t direction = 1;
            size_t layers = 1;
            auto dlc = weights_layer->get_shape()[1] / (lstm_n_gates * direction * layers);
            auto slc = weights_layer->get_shape()[0];
            auto dic = weights_iter->get_shape()[1] / (lstm_n_gates * direction * layers);
            auto sic = weights_iter->get_shape()[0];

            if (dlc != dic)
            {
                NGRAPH_DEBUG << "Not fusing, since Lstm kernel requires dst_layer feature size "
                             << "equals to dts_iter feature size";
                return false;
            }

            std::shared_ptr<Node> src_iter =
                std::make_shared<op::Concat>(NodeVector{hidden_state, cell_state}, 0);
            if (src_layer->get_shape()[1] != slc || src_iter->get_shape()[1] != sic)
            {
                NGRAPH_DEBUG << "Feature size mismatch between weights and input tensors";
                return false;
            }

            auto bias = std::make_shared<op::Add>(pattern_map[bias_i2h], pattern_map[bias_h2h]);

            auto lstm_node =
                std::make_shared<op::Lstm>(src_layer, src_iter, weights_layer, weights_iter, bias);

            auto lstm_ht_output = std::make_shared<op::GetOutputElement>(lstm_node, 0);
            auto lstm_ht_ct_output = std::make_shared<op::GetOutputElement>(lstm_node, 1);

            // dst_iter of lstm mkldnn output holds the results of both recurrent state
            // tensor outputs. we need to slice the ct.
            auto ht_slice = std::make_shared<op::Slice>(
                lstm_ht_output, Coordinate{0, 0}, Coordinate{batch_size, dlc});
            auto ct_slice = std::make_shared<op::Slice>(
                lstm_ht_ct_output, Coordinate{batch_size, 0}, Coordinate{(2 * batch_size), dic});

            if (lstm_node->get_outputs().at(0).get_inputs().size() != 2)
            {
                throw ngraph_error("Lstm node doesnt have two outputs");
            }
            // Now identify the nodes which consumes the output of LSTM nodes
            // and replace them accordingly
            // find the user's for {ht|ct} and replace them with lstm_goe_1
            if (ngraph::is_used(pattern_map[ct_label].get()))
            {
                replace_collapse_node_user(pattern_map[ct_label], ct_slice->get_outputs().at(0));
            }
            // find the user's for {ht} and replace them with lstm_goe_0
            ngraph::replace_node(m.get_match_root(), ht_slice);
            return true;
        };
    auto m = std::make_shared<pattern::Matcher>(ht, callback);
    this->add_matcher(m);
}

void ngraph::runtime::cpu::pass::RNNFusion::construct_rnn_lstm_fprop()
{
    auto src_layer_label = std::make_shared<pattern::op::Label>(element::f32, Shape{30, 100});

    auto ht_label = std::make_shared<pattern::op::Label>(element::f32, Shape{10, 100});
    auto ct_label = std::make_shared<pattern::op::Label>(element::f32, Shape{10, 100});
    auto recurrent_inputs = std::make_shared<op::Concat>(NodeVector{ht_label, ct_label}, 0);
    auto src_iter_label = std::make_shared<pattern::op::Label>(
        recurrent_inputs, nullptr, NodeVector{recurrent_inputs});

    auto weights_i2h_param = std::make_shared<pattern::op::Label>(
        element::f32, Shape{400, 100}, pattern::has_class<op::Parameter>());
    auto weights_i2h_reshape =
        std::make_shared<op::Reshape>(weights_i2h_param, AxisVector{1, 0}, Shape{100, 400});
    auto weights_i2h_label = std::make_shared<pattern::op::Label>(
        weights_i2h_reshape, nullptr, NodeVector{weights_i2h_reshape});

    auto weights_h2h_param = std::make_shared<pattern::op::Label>(
        element::f32, Shape{400, 100}, pattern::has_class<op::Parameter>());
    auto weights_h2h_reshape =
        std::make_shared<op::Reshape>(weights_h2h_param, AxisVector{1, 0}, Shape{100, 400});
    auto weights_h2h_label = std::make_shared<pattern::op::Label>(
        weights_h2h_reshape, nullptr, NodeVector{weights_h2h_reshape});

    auto bias_i2h = std::make_shared<pattern::op::Label>(element::f32, Shape{400});
    auto bias_h2h = std::make_shared<pattern::op::Label>(element::f32, Shape{400});
    auto add_bias = std::make_shared<op::Add>(bias_i2h, bias_h2h);
    auto bias_label = std::make_shared<pattern::op::Label>(add_bias, nullptr, NodeVector{add_bias});

    auto lstm = std::make_shared<op::Lstm>(
        src_layer_label, src_iter_label, weights_i2h_label, weights_h2h_label, bias_label);
    auto lstm_goe = std::make_shared<op::GetOutputElement>(lstm, 1);
    auto lstm_goe_label =
        std::make_shared<pattern::op::Label>(lstm_goe, nullptr, NodeVector{lstm_goe});
    auto lstm_goe_slice =
        std::make_shared<op::Slice>(lstm_goe_label, Coordinate{10, 0}, Coordinate{20, 100});

    pattern::recurrent_graph_rewrite_callback callback = [ht_label,
                                                          ct_label,
                                                          weights_i2h_param,
                                                          weights_h2h_param,
                                                          bias_i2h,
                                                          bias_h2h,
                                                          lstm_goe_label,
                                                          src_layer_label,
                                                          src_iter_label,
                                                          weights_i2h_label,
                                                          weights_h2h_label,
                                                          bias_label](
        pattern::RecurrentMatcher& m) {

        NGRAPH_DEBUG << " In recurrent RNN fusion callback";

        auto concat_rnn_inputs_across_timestep =
            [&](std::shared_ptr<pattern::op::Label> input_label) -> std::shared_ptr<Node> {
            NodeVector concat_args;
            // src_layer -> concatenate input symbols from different LSTM cells belonging to same RNN layer
            // in the order 0, 1, 2... t time slice
            {
                auto node_labels = m.get_bound_nodes_for_pattern(input_label);
                if (node_labels.size() > 1)
                {
                    std::reverse(node_labels.begin(), node_labels.end());
                    return std::make_shared<op::Concat>(node_labels, 0);
                }
                else
                {
                    return node_labels[0];
                }
            }
        };

        auto src_layer = concat_rnn_inputs_across_timestep(src_layer_label);
        auto src_iter_bounded_nodes = m.get_bound_nodes_for_pattern(src_iter_label);
        auto src_iter = src_iter_bounded_nodes[src_iter_bounded_nodes.size() - 1];

        auto weights_layer = m.get_bound_nodes_for_pattern(weights_i2h_label)[0];
        auto weights_iter = m.get_bound_nodes_for_pattern(weights_h2h_label)[0];
        auto bias = m.get_bound_nodes_for_pattern(bias_label)[0];

        auto num_of_lstm_matched = m.get_number_of_recurrent_matches();
        size_t lstm_n_gates = 4;
        // TODO: assert for batch_size, sequence length and num_of_lstm's fused
        size_t batch_size = src_layer->get_shape()[0] / num_of_lstm_matched;
        size_t sequence_len = num_of_lstm_matched;
        size_t src_layer_feature_size = weights_layer->get_shape()[0];
        size_t dlc = weights_layer->get_shape()[1] / lstm_n_gates;
        size_t dic = weights_iter->get_shape()[1] / lstm_n_gates;
        size_t src_iter_feature_size = weights_iter->get_shape()[0];
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
        NGRAPH_DEBUG << "src_iter_feature_size: " << src_iter_feature_size;

        // if we have have not found all the LSTM cells belonging to a layer
        // will return safely
        std::shared_ptr<Node> src_iter_arg = src_iter->get_arguments()[0];
        if (!(std::dynamic_pointer_cast<op::Broadcast>(src_iter_arg) &&
              std::dynamic_pointer_cast<op::Constant>(src_iter_arg->get_argument(0))) &&
            !(std::dynamic_pointer_cast<op::Constant>(src_iter_arg)))
        {
            return false;
        }

        if ((src_layer->get_shape()[0] / batch_size) != sequence_len)
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
                "Pattern matcher error src_layer, src_iter, weights_layer, weighst_iter should "
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
                                             lstm_n_gates,
                                             sequence_len,
                                             num_cell_states,
                                             direction,
                                             num_fused_rnn_layers);

        std::vector<std::shared_ptr<op::Slice>> ht_slice_per_timestep(num_of_lstm_matched, nullptr);
        auto rnn_ht_goe = std::make_shared<op::GetOutputElement>(rnn, 0);
        auto rnn_ht_ct_out = std::make_shared<op::GetOutputElement>(rnn, 1);

        // capture the slices in the reverse order, so it corrosponds to lstm_goes order captured by the Pattern matcher
        // slice the rnn ht's
        size_t start_index = 0;
        size_t end_index = batch_size;
        for (size_t i = 0; i < num_of_lstm_matched; i++)
        {
            ht_slice_per_timestep[i] =
                (std::make_shared<op::Slice>(rnn_ht_goe,
                                             Coordinate{start_index, 0},
                                             Coordinate{end_index, src_iter_feature_size}));
            start_index += batch_size;
            end_index += batch_size;
        }
        std::reverse(ht_slice_per_timestep.begin(), ht_slice_per_timestep.end());

        NGRAPH_DEBUG << "rnn_time_slice: " << ht_slice_per_timestep.size();

        // find the lstm's nodes captured in PM
        auto lstm_goes = m.get_bound_nodes_for_pattern(lstm_goe_label);
        std::vector<std::shared_ptr<ngraph::Node>> lstm_nodes;

        // we need to collect LSTM from GOE's, in order to deterministicaly determine
        // the individaual time slice output ht. lstm_goes will hold the GOE in the decreasing
        // order of the time slices
        for (size_t i = 0; i < lstm_goes.size(); i++)
        {
            // lstm's will be the input to GOE's
            lstm_nodes.push_back(lstm_goes[i]->get_arguments()[0]);
        }

        // collect all the consumers of LSTM goe's (ht)
        std::set<std::shared_ptr<ngraph::Node>> lstm_goe0_user;
        std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node>>
            map_goe0_user_to_lstm_slices;
        std::shared_ptr<Node> goe_0;

        for (size_t index = 0; index < lstm_nodes.size(); index++)
        {
            auto goe_nodes = op::get_output_elements(lstm_nodes[index]);

            // if their is no GOE followed by the Lstm, their might be pattern match error
            // we will return safely
            if (goe_nodes.size() == 0)
            {
                return false;
            }

            // dst_layer of the lstm cell
            auto goe_0 = goe_nodes[0];

            // dst_iter of the lstm cell
            auto goe_1 = goe_nodes[1];

            if (goe_0)
            {
                for (auto goe0_user : goe_0->get_users())
                {
                    if (ngraph::is_used(goe0_user.get()))
                    {
                        lstm_goe0_user.insert(goe0_user);
                        map_goe0_user_to_lstm_slices.insert(
                            make_pair(goe0_user, ht_slice_per_timestep[index]));

                        NGRAPH_DEBUG << "ht_slice: " << ht_slice_per_timestep[index]->get_name()
                                     << " goe0_user " << goe0_user->get_name() << " ";
                    }
                }
            }

            // we need to only check the last LSTM cell Ct user and replace if needed.
            if ((index == 0) && goe_1)
            {
                // dst_iter of lstm mkldnn output holds the results of both recurrent state
                // tensor outputs. we will replace the GOE, since RNN->GOE1 and LSTM_n->GOE1
                // holds the same output
                replace_collapse_node_user(goe_1, rnn_ht_ct_out->get_outputs().at(0));
            }
        }

        // now go through the lstm goe_0 consumers and replace them with the slice
        for (auto& node : lstm_goe0_user)
        {
            if (ngraph::is_used(node.get()))
            {
                if (std::dynamic_pointer_cast<op::Slice>(node))
                {
                    ngraph::replace_node(node, map_goe0_user_to_lstm_slices[node]);
                }
                else
                {
                    throw ngraph_error(
                        "We can replace Rnn->Goe_0->Slice only on Lstm->Goe_0->Slice");
                }
            }
        }
        NGRAPH_DEBUG << "End of recurrent fusion call back "
                     << "matched_node: " << m.get_match_root()->get_name();
        return true;

    };

    auto m = std::make_shared<pattern::RecurrentMatcher>(
        lstm_goe_slice,
        ct_label,
        std::set<std::shared_ptr<pattern::op::Label>>{
            weights_i2h_param, weights_h2h_param, bias_i2h, bias_h2h},
        callback);
    this->add_matcher(m);
}

static std::shared_ptr<Node> stack_rnn_inputs(NodeVector& rnn_input_nodes)
{
    std::reverse(rnn_input_nodes.begin(), rnn_input_nodes.end());
    return std::make_shared<op::Concat>(rnn_input_nodes, 0);
}

void ngraph::runtime::cpu::pass::MultiLayerRNNFusion::construct_multi_layer_rnn_fusion_fprop()
{
    auto src_layer_label = std::make_shared<pattern::op::Label>(element::f32, Shape{30, 100});
    auto src_iter_label = std::make_shared<pattern::op::Label>(element::f32, Shape{20, 100});
    auto weights_layer_label = std::make_shared<pattern::op::Label>(element::f32, Shape{100, 400});
    auto weights_iter_label = std::make_shared<pattern::op::Label>(element::f32, Shape{100, 400});
    auto bias_label = std::make_shared<pattern::op::Label>(element::f32, Shape{400});

    size_t ref_number_of_timesteps = 3;
    size_t ref_number_of_gates_per_cell = 4;
    size_t ref_src_seq_length = 3;
    size_t ref_num_rnn_cell_states = 2;
    size_t ref_rnn_direction = 1;
    size_t ref_num_of_rnn_fused_layer = 1;

    auto ref_rnn_node = std::make_shared<op::Rnn>(src_layer_label,
                                                  src_iter_label,
                                                  weights_layer_label,
                                                  weights_iter_label,
                                                  bias_label,
                                                  ref_number_of_timesteps,
                                                  ref_number_of_gates_per_cell,
                                                  ref_src_seq_length,
                                                  ref_num_rnn_cell_states,
                                                  ref_rnn_direction,
                                                  ref_num_of_rnn_fused_layer);

    auto rnn_goe0 = std::make_shared<op::GetOutputElement>(ref_rnn_node, 0);

    auto rnn_goe0_label =
        std::make_shared<pattern::op::Label>(rnn_goe0, nullptr, NodeVector{rnn_goe0});

    pattern::recurrent_graph_rewrite_callback callback = [src_layer_label,
                                                          src_iter_label,
                                                          weights_layer_label,
                                                          weights_iter_label,
                                                          bias_label,
                                                          rnn_goe0_label](
        pattern::RecurrentMatcher& m) {
        if (m.get_number_of_recurrent_matches() <= 1)
        {
            return false;
        }
        auto src_nodes = m.get_bound_nodes_for_pattern(src_layer_label);
        auto rnn_output_recurrent_hidden_state = m.get_bound_nodes_for_pattern(rnn_goe0_label);
        auto number_of_rnn_cell_matched = m.get_number_of_recurrent_matches();

        NGRAPH_DEBUG << " In Recurrent multi layer RNN fusion callback ";
        NGRAPH_DEBUG << " Number of RNN's Matched: " << number_of_rnn_cell_matched;
        NGRAPH_DEBUG << " matched_root: " << m.get_match_root()->get_name();

        std::shared_ptr<Node> src_layer = nullptr;
        auto src_layer_bounded_nodes = m.get_bound_nodes_for_pattern(src_layer_label);
        auto src_iter_bounded_nodes = m.get_bound_nodes_for_pattern(src_iter_label);
        auto weights_layer_bounded_nodes = m.get_bound_nodes_for_pattern(weights_layer_label);
        auto weights_iter_bounded_nodes = m.get_bound_nodes_for_pattern(weights_iter_label);
        auto bias_bounded_nodes = m.get_bound_nodes_for_pattern(bias_label);
        auto rnn_goe0_bounded_nodes = m.get_bound_nodes_for_pattern(rnn_goe0_label);

        std::vector<std::shared_ptr<op::Rnn>> rnn_nodes;
        for (size_t index = 0; index < rnn_goe0_bounded_nodes.size(); index++)
        {
            if (auto rnn_op = std::dynamic_pointer_cast<op::Rnn>(
                    rnn_goe0_bounded_nodes[index]->get_arguments()[0]))
            {
                // we will look at the matched RNN cells and only fuse the RNN if we have
                // SLC == DLC
                if (rnn_op->get_dst_layer_feature_size() == rnn_op->get_src_layer_feature_size())
                {
                    rnn_nodes.push_back(rnn_op);
                }
                else
                {
                    return false;
                }
            }
            else
            {
                NGRAPH_DEBUG << "PM error, input to GOE is not RNN";
                return false;
            }
        }

        // we will return if we have less than two nodes to fuse
        if (rnn_nodes.size() <= 1)
        {
            return false;
        }
        // the last matched rnn cell with slc=dlc will be in the input to the new fused
        // node, PM captures the RNN cell in the reverse order.
        // {RNN7, RNN6, RNN5.....RNN0}
        src_layer = src_layer_bounded_nodes[src_layer_bounded_nodes.size() - 1];
        auto src_iter = stack_rnn_inputs(src_iter_bounded_nodes);
        auto weights_layer = stack_rnn_inputs(weights_layer_bounded_nodes);
        auto weights_iter = stack_rnn_inputs(weights_iter_bounded_nodes);
        auto bias = stack_rnn_inputs(bias_bounded_nodes);

        size_t num_time_steps = rnn_nodes[0]->get_num_timesteps();
        size_t lstm_n_gates = rnn_nodes[0]->get_gates_per_cell();
        size_t batch_size = rnn_nodes[0]->get_batch_size();
        size_t sequence_len = rnn_nodes[0]->get_src_sequence_length();
        size_t src_layer_feature_size = rnn_nodes[0]->get_src_layer_feature_size();
        size_t feature_size = rnn_nodes[0]->get_src_iter_feature_size();
        size_t num_rnn_cell_states = rnn_nodes[0]->get_num_cell_states();
        size_t rnn_direction = rnn_nodes[0]->get_direction();
        size_t num_fused_rnn_layers = rnn_nodes.size();

        NGRAPH_DEBUG << "src_layer: " << join(src_layer->get_shape());
        NGRAPH_DEBUG << "src_iter: " << join(src_iter->get_shape());
        NGRAPH_DEBUG << "weights_layer: " << join(weights_layer->get_shape());
        NGRAPH_DEBUG << "weights_iter: " << join(weights_iter->get_shape());
        NGRAPH_DEBUG << "bias: " << join(bias->get_shape());
        NGRAPH_DEBUG << "src_seq_len: " << sequence_len;
        NGRAPH_DEBUG << "batch_size: " << batch_size;
        NGRAPH_DEBUG << "feature_size: " << feature_size;

        if ((src_layer->get_shape()[0] / batch_size) != rnn_nodes[0]->get_num_timesteps())
        {
            throw ngraph_error(
                " input symbols for the layer fused RNN op, should be captured only for the first "
                "layer");
        }

        if ((src_iter->get_arguments().size()) != num_fused_rnn_layers)
        {
            throw ngraph_error(
                "number of states(ht_1|ct_1) for RNN op in the layer fusion is not equal to num of "
                "fused_rnn_layers");
        }

        if ((weights_layer->get_arguments().size()) != num_fused_rnn_layers)
        {
            throw ngraph_error(
                "weights w.r.to input symbols of RNN op in the layer fusion is not equal to num of "
                "fused_rnn_layers");
        }

        if ((weights_iter->get_arguments().size()) != num_fused_rnn_layers)
        {
            throw ngraph_error(
                "weights w.r.to cell states of RNN op in the layer fusion is not equal to num of "
                "fused_rnn_layers");
        }

        if ((bias->get_arguments().size()) != num_fused_rnn_layers)
        {
            throw ngraph_error(
                "bias of RNN op in the layer fusion is not equal to num of fused_rnn_layers");
        }

        auto rnn = std::make_shared<op::Rnn>(src_layer,
                                             src_iter,
                                             weights_layer,
                                             weights_iter,
                                             bias,
                                             num_time_steps,
                                             lstm_n_gates,
                                             sequence_len,
                                             num_rnn_cell_states,
                                             rnn_direction,
                                             num_fused_rnn_layers);

        auto layer_rnn_ht = std::make_shared<op::GetOutputElement>(rnn, 0);
        auto layer_rnn_ht_ct = std::make_shared<op::GetOutputElement>(rnn, 1);

        // Replace all the users of RNN cell state {ct} across different user.
        auto replace_rnn_output_cellstate = [&](std::shared_ptr<Node> rnn_ct_goe1, size_t layer) {

            // multi layerd fused rnn second output {GOE1} holds the recurrent output state tensors for the last cell
            // of all the layers, {{ht_1 | ct_1} || {ht2 |ct2} || ....{htn | ctn}}
            // we will slice the cell state output tensor {ct_*} from the fused RNN kerenel output and feeds
            // {ct_*} consumer if any
            auto ct_slice = std::make_shared<op::Slice>(
                layer_rnn_ht_ct,
                Coordinate{static_cast<unsigned long>(
                               ((layer - 1) * batch_size * num_rnn_cell_states) + batch_size),
                           0},
                Coordinate{static_cast<unsigned long>(num_rnn_cell_states * layer * batch_size),
                           static_cast<unsigned long>(feature_size)});

            replace_collapse_node_user(rnn_ct_goe1, ct_slice->get_outputs().at(0));
        };

        // we will replace cell_state {ct} of all the matched RNN cell
        // with the new {ct} of the fused RNN cell
        // Note: RNN cells are captured in the reverse order
        // i.e {RNN7, RNN6, RNN5.... RNN0}
        for (size_t index = 0; index < rnn_nodes.size(); index++)
        {
            auto goe_nodes = op::get_output_elements(rnn_nodes[index]);
            // if their is no GOE followed by the Lstm, their might be pattern match error
            // we will return safely
            if (goe_nodes.size() == 0)
            {
                return false;
            }

            // dst_layer of the lstm cell
            auto goe_0 = goe_nodes[0];
            // dst_iter of the lstm cell
            auto goe_1 = goe_nodes[1];

            if (goe_1)
            {
                int layer_index = num_fused_rnn_layers - index;
                replace_rnn_output_cellstate(goe_1, layer_index);
            }

            // dst_layer of layer fused rnn holds the intermediate results of all the lstm cells
            // belonging to the last layer we will replace the GOE, since RNN_n->GOE0 and MutliLayerRnn->GOE0
            // holds the same output
            if ((index == 0) && goe_0)
            {
                replace_collapse_node_user(goe_0, layer_rnn_ht->get_outputs().at(0));
            }
        }
        return true;
    };

    std::set<std::shared_ptr<pattern::op::Label>> empty_correlated_matches;
    auto m = std::make_shared<pattern::RecurrentMatcher>(
        rnn_goe0_label, src_layer_label, empty_correlated_matches, callback);
    this->add_matcher(m);
}

void ngraph::runtime::cpu::pass::BiDirectionalRnn::construct_bidirectional_rnn()
{
    auto rnn_left_to_right = std::make_shared<pattern::op::Label>(
        element::f32, Shape{1, 256}, pattern::has_class<op::Rnn>());
    auto rnn_right_to_left = std::make_shared<pattern::op::Label>(
        element::f32, Shape{1, 256}, pattern::has_class<op::Rnn>());

    auto reshape_pred = [](std::shared_ptr<Node> n) {
        return (std::dynamic_pointer_cast<op::Reshape>(n) != nullptr);
    };
    auto rnn_left_to_right_goe0 = std::make_shared<op::GetOutputElement>(rnn_left_to_right, 0);
    auto rnn_right_to_left_goe0 = std::make_shared<op::GetOutputElement>(rnn_right_to_left, 0);

    auto rnn_rtol_goe0_reshape =
        std::make_shared<pattern::op::Skip>(rnn_right_to_left_goe0, reshape_pred);
    auto rnn_ltor_goe0_reshape =
        std::make_shared<pattern::op::Skip>(rnn_left_to_right_goe0, reshape_pred);

    auto sequence_len = std::make_shared<pattern::op::Label>(element::f32, Shape{1});
    auto reverse_seq =
        std::make_shared<op::ReverseSequence>(rnn_rtol_goe0_reshape, sequence_len, 0, 0);
    auto concat = std::make_shared<op::Concat>(NodeVector{rnn_ltor_goe0_reshape, reverse_seq}, 0);

    // Define a call back that needs to called once the DFG matches the pattern
    ngraph::pattern::graph_rewrite_callback callback = [sequence_len,
                                                        rnn_left_to_right,
                                                        rnn_right_to_left](pattern::Matcher& m) {

        auto pattern_map = m.get_pattern_map();
        auto rnn_ltor_node = std::dynamic_pointer_cast<op::Rnn>(pattern_map[rnn_left_to_right]);
        auto rnn_rtol_node = std::dynamic_pointer_cast<op::Rnn>(pattern_map[rnn_right_to_left]);

        if (rnn_ltor_node->get_src_sequence_length() != rnn_rtol_node->get_src_sequence_length())
        {
            NGRAPH_DEBUG << " Not fusing, timestep of rnn's in both direction should match";
            return false;
        }

        if (rnn_ltor_node->get_src_layer_feature_size() !=
            rnn_rtol_node->get_src_layer_feature_size())
        {
            NGRAPH_DEBUG << " Not fusing, feature_size of rnn's in both direction should match";
            return false;
        }

        if (rnn_ltor_node->get_src_iter_feature_size() !=
            rnn_rtol_node->get_src_iter_feature_size())
        {
            NGRAPH_DEBUG << " Not fusing, feature_size of rnn's in both direction should match";
            return false;
        }

        if (rnn_ltor_node->get_batch_size() != rnn_rtol_node->get_batch_size())
        {
            NGRAPH_DEBUG << " Not fusing, feature_size of rnn's in both direction should match";
            return false;
        }

        size_t num_time_steps = rnn_ltor_node->get_num_timesteps();
        size_t lstm_n_gates = rnn_ltor_node->get_gates_per_cell();
        size_t batch_size = rnn_ltor_node->get_batch_size();
        size_t sequence_len = rnn_ltor_node->get_src_sequence_length();
        size_t num_rnn_cell_states = rnn_ltor_node->get_num_cell_states();
        size_t rnn_direction = 2;
        size_t num_fused_rnn_layers = 1;

        auto construct_birnn_inputs = [&](int index) {

            auto nodes =
                NodeVector{rnn_ltor_node->get_argument(index), rnn_rtol_node->get_argument(index)};
            return std::make_shared<op::Concat>(nodes, 0);
        };

        auto src_layer = rnn_ltor_node->get_arguments()[0];
        auto src_iter = construct_birnn_inputs(1);
        auto weights_layer = construct_birnn_inputs(2);
        auto weights_iter = construct_birnn_inputs(3);
        auto bias = construct_birnn_inputs(4);

        auto rnn = std::make_shared<op::Rnn>(src_layer,
                                             src_iter,
                                             weights_layer,
                                             weights_iter,
                                             bias,
                                             num_time_steps,
                                             lstm_n_gates,
                                             sequence_len,
                                             num_rnn_cell_states,
                                             rnn_direction,
                                             num_fused_rnn_layers);

        auto layer_rnn_ht = std::make_shared<op::GetOutputElement>(rnn, 0);
        std::cout << "In bi Rnn call back" << std::endl;
        ngraph::replace_node(m.get_match_root(), layer_rnn_ht);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(concat, callback);
    this->add_matcher(m);
}
