//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "gpu_rnn_fusion.hpp"
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
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"
#include "ngraph/runtime/gpu/op/rnn.hpp"

#define RETURN_IF_FALSE(cond, message)                                                             \
    if (!(cond))                                                                                   \
    {                                                                                              \
        NGRAPH_DEBUG << message;                                                                   \
        return false;                                                                              \
    }

using namespace ngraph;
void ngraph::runtime::gpu::pass::LSTMFusion::construct_sigmoid()
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
    auto callback = [input](pattern::Matcher& m) {
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

    auto m = std::make_shared<ngraph::pattern::Matcher>(divide_1_over_exp);
    this->add_matcher(m, callback);
}

static std::shared_ptr<Node> compute_lstm_params(const std::shared_ptr<Node>& w_x,
                                                 const std::shared_ptr<Node>& w_h,
                                                 const std::shared_ptr<Node>& b_x,
                                                 const std::shared_ptr<Node>& b_h)
{
    // check if concat of params exists already
    // if so, use it
    for (auto& node : w_x->get_users())
    {
        for (auto& possible_concat : node->get_users())
        {
            if (auto concat = std::dynamic_pointer_cast<op::Concat>(possible_concat))
            {
                return concat;
            }
        }
    }

    NodeVector flat_params;
    for (auto& param : NodeVector{w_x, w_h, b_x, b_h})
    {
        auto shape = param->get_shape();
        flat_params.push_back(std::make_shared<op::Reshape>(
            param, get_default_order(shape), Shape{shape_size(shape)}));
    }
    return std::make_shared<op::Concat>(flat_params, 0);
}

void ngraph::runtime::gpu::pass::LSTMFusion::construct_lstm_fprop()
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

    // ct-1 -> cell state
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

    // Define a call back that needs to called once the DFG matches the pattern
    auto callback = [ct_label,
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

        RETURN_IF_FALSE(bias_i2h->get_shape().size() == 1 && bias_h2h->get_shape().size() == 1,
                        "Bias should have rank of 1 for Rnn op");

        // Determine which is ht_1 and xt. but if both xt and ht_1 have the same shape we need to
        // capture this reliably in the RNN fusion.
        std::shared_ptr<op::gpu::Rnn> lstm = nullptr;
        bool intermediate_lstm = false;
        if (std::dynamic_pointer_cast<op::GetOutputElement>(pattern_map[ct_1]))
        {
            intermediate_lstm = true;
        }
        // if the matched LSTM is the first cell we need to check if symbol input_xt corresponds
        // to the input data tensor, or the hidden (recurrent) data tensor
        if (!intermediate_lstm &&
            (std::dynamic_pointer_cast<op::Broadcast>(pattern_map[hidden_ht]) &&
             std::dynamic_pointer_cast<op::Constant>(pattern_map[hidden_ht]->get_argument(0))))
        // label input_xt is the input data to the first LSTM
        {
            auto params = compute_lstm_params(pattern_map[weights_i2h],
                                              pattern_map[weights_h2h],
                                              pattern_map[bias_i2h],
                                              pattern_map[bias_h2h]);
            lstm = std::make_shared<op::gpu::Rnn>(pattern_map[input_xt],
                                                  pattern_map[hidden_ht],
                                                  params,
                                                  pattern_map[ct_1],
                                                  1,
                                                  4,
                                                  1,
                                                  pattern_map[input_xt]->get_shape()[1],
                                                  pattern_map[hidden_ht]->get_shape()[1],
                                                  1,
                                                  1);
        }
        else if (!intermediate_lstm &&
                 (std::dynamic_pointer_cast<op::Broadcast>(pattern_map[input_xt]) &&
                  std::dynamic_pointer_cast<op::Constant>(pattern_map[input_xt]->get_argument(0))))
        // label hidden_ht is the input data to the first LSTM
        {
            auto params = compute_lstm_params(pattern_map[weights_h2h],
                                              pattern_map[weights_i2h],
                                              pattern_map[bias_h2h],
                                              pattern_map[bias_i2h]);
            lstm = std::make_shared<op::gpu::Rnn>(pattern_map[hidden_ht],
                                                  pattern_map[input_xt],
                                                  params,
                                                  pattern_map[ct_1],
                                                  1,
                                                  4,
                                                  1,
                                                  pattern_map[hidden_ht]->get_shape()[1],
                                                  pattern_map[input_xt]->get_shape()[1],
                                                  1,
                                                  1);
        }
        else if (pattern_map[hidden_ht]->get_arguments().size() &&
                 pattern_map[ct_1]->get_arguments().at(0)->get_instance_id() ==
                     pattern_map[hidden_ht]->get_arguments().at(0)->get_instance_id())
        // this still has a bug vector: if the hidden input ht is a non-broadcasted constant
        // it will be misclassified as input data xt
        {
            // label input_xt is the output data from the previous LSTM cell
            NGRAPH_DEBUG << "ct_shape : " << join(pattern_map[ct_1]->get_shape())
                         << " hidden state shape: " << join(pattern_map[hidden_ht]->get_shape());
            auto params = compute_lstm_params(pattern_map[weights_i2h],
                                              pattern_map[weights_h2h],
                                              pattern_map[bias_i2h],
                                              pattern_map[bias_h2h]);
            lstm = std::make_shared<op::gpu::Rnn>(pattern_map[input_xt],
                                                  pattern_map[hidden_ht],
                                                  params,
                                                  pattern_map[ct_1],
                                                  1,
                                                  4,
                                                  1,
                                                  pattern_map[input_xt]->get_shape()[1],
                                                  pattern_map[hidden_ht]->get_shape()[1],
                                                  1,
                                                  1);
        }
        else
        {
            // label hidden_ht is the output data from the previous LSTM cell
            NGRAPH_DEBUG << "ct_shape: " << join(pattern_map[ct_1]->get_shape())
                         << " hidden state shape: " << join(pattern_map[input_xt]->get_shape());
            auto params = compute_lstm_params(pattern_map[weights_h2h],
                                              pattern_map[weights_i2h],
                                              pattern_map[bias_h2h],
                                              pattern_map[bias_i2h]);
            lstm = std::make_shared<op::gpu::Rnn>(pattern_map[hidden_ht],
                                                  pattern_map[input_xt],
                                                  params,
                                                  pattern_map[ct_1],
                                                  1,
                                                  4,
                                                  1,
                                                  pattern_map[hidden_ht]->get_shape()[1],
                                                  pattern_map[input_xt]->get_shape()[1],
                                                  1,
                                                  1);
        }

        auto ht_output = std::make_shared<op::GetOutputElement>(lstm, 0);
        auto ct_output = std::make_shared<op::GetOutputElement>(lstm, 2);

        // Now identify the nodes which consume the outputs of LSTM nodes
        // and replace them accordingly
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
    auto m = std::make_shared<pattern::Matcher>(ht);
    this->add_matcher(m, callback);
}

static std::shared_ptr<ngraph::Node>
    compute_rnn_args(std::vector<std::shared_ptr<pattern::op::Label>>& rnn_labels,
                     pattern::RecurrentMatcher& m,
                     bool concat_all = false)
{
    NGRAPH_DEBUG << "Inside compute arg " << rnn_labels.size();

    // src_layer -> concatenate input symbols from different LSTM cells belonging to same RNN layer
    // in the order 0, 1, 2... t time slice
    if (concat_all)
    {
        auto node_labels = m.get_bound_nodes_for_pattern(rnn_labels[0]);
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
    // i2h or h2h weights shared between LSTM cells
    else
    {
        // return the first instance of the weight matrix for this RNN layer
        // weights and biases are reused for all cells in a layer.
        auto node_labels = m.get_bound_nodes_for_pattern(rnn_labels[0]);
        return node_labels[node_labels.size() - 1];
    }
}

void ngraph::runtime::gpu::pass::RNNFusion::construct_rnn_lstm_fprop()
{
    auto xt = std::make_shared<pattern::op::Label>(element::f32, Shape{32, 100});
    auto ht_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{32, 100});
    auto params_label = std::make_shared<pattern::op::Label>(
        element::f32, Shape{400 * 100 + 400 * 100 + 400 + 400});
    auto rpattern_ct_1 = std::make_shared<pattern::op::Label>(element::f32, Shape{32, 100});

    auto lstm = std::make_shared<op::gpu::Rnn>(xt,
                                               ht_1,
                                               params_label,
                                               rpattern_ct_1,
                                               1,
                                               4,
                                               1,
                                               xt->get_shape()[1],
                                               ht_1->get_shape()[1],
                                               1,
                                               1);
    auto goe = std::make_shared<op::GetOutputElement>(lstm, 0); // hidden output
    auto lstm_node_label = std::make_shared<pattern::op::Label>(goe, nullptr, NodeVector{goe});

    auto callback = [lstm_node_label, xt, ht_1, params_label, rpattern_ct_1](
        pattern::RecurrentMatcher& m) {

        NGRAPH_DEBUG << " In RNN fusion callback";

        auto ht_1_label = m.get_bound_nodes_for_pattern(ht_1);
        auto params_bound = m.get_bound_nodes_for_pattern(params_label);

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
        // here xt is determined to be the hidden (recurrent) input data and so ht is the
        // feedforward input
        {
            // concatenate the sequence inputs for a given layer
            std::vector<std::shared_ptr<pattern::op::Label>> src_layer_labels{ht_1};
            src_layer = compute_rnn_args(src_layer_labels, m, true);

            // concatenate the hidden (recurrent) input with the cell
            std::vector<std::shared_ptr<pattern::op::Label>> src_iter_labels{xt};
            src_iter = compute_rnn_args(src_iter_labels, m);
        }
        else if (std::dynamic_pointer_cast<op::Broadcast>(
                     hidden_ht_array[hidden_ht_array.size() - 1]) &&
                 std::dynamic_pointer_cast<op::Constant>(
                     hidden_ht_array[hidden_ht_array.size() - 1]->get_argument(0)))
        // here ht is determined to be the hidden (recurrent) input data and so xt is the
        // feedforward input
        {
            std::vector<std::shared_ptr<pattern::op::Label>> src_layer_labels{xt};
            src_layer = compute_rnn_args(src_layer_labels, m, true);

            std::vector<std::shared_ptr<pattern::op::Label>> src_iter_labels{ht_1};
            src_iter = compute_rnn_args(src_iter_labels, m);
        }
        else
        {
            // dont fuse, if the PM didn't discover all the cells belonging to RNN layer.
            // we dont want to throw an assertion, if pattern matcher cannot discover all
            // nodes belonging to RNN, instead we will return and can compute LSTM cell wise
            return false;
        }

        std::vector<std::shared_ptr<pattern::op::Label>> params_labels{params_label};
        auto params = compute_rnn_args(params_labels, m);

        std::vector<std::shared_ptr<pattern::op::Label>> state_iter_labels{rpattern_ct_1};
        auto state_iter = compute_rnn_args(state_iter_labels, m);

        auto num_of_lstm_matched = m.get_number_of_recurrent_matches();
        if (num_of_lstm_matched <= 1)
        {
            return false;
        }

        size_t num_gates_in_lstm = 4;
        size_t batch_size = src_layer->get_shape()[0] / num_of_lstm_matched;
        size_t sequence_len = num_of_lstm_matched;
        size_t src_layer_feature_size = src_layer->get_shape()[1];
        size_t feature_size = ht_1_label[0]->get_shape()[1];
        // number of states for LSTM is 2
        size_t direction = 1;
        size_t num_fused_rnn_layers = 1;

        NGRAPH_DEBUG << "src_layer: " << join(src_layer->get_shape());
        NGRAPH_DEBUG << "src_iter: " << join(src_iter->get_shape());
        NGRAPH_DEBUG << "src_seq_len: " << sequence_len;
        NGRAPH_DEBUG << "batch_size: " << batch_size;
        NGRAPH_DEBUG << "feature_size: " << feature_size;

        RETURN_IF_FALSE(src_layer->get_arguments().size() == sequence_len ||
                            std::dynamic_pointer_cast<op::Parameter>(src_layer),
                        "number of lstm inputs captured in the RNN fusion is not equal to "
                        "src_sequence_length");
        RETURN_IF_FALSE(!std::dynamic_pointer_cast<op::Parameter>(src_layer) || sequence_len == 1,
                        "number of lstm inputs captured in the RNN fusion is not equal to "
                        "src_sequence_length");

        auto src_layer_rank = src_layer->get_shape().size();
        auto src_iter_rank = src_iter->get_shape().size();
        RETURN_IF_FALSE(src_layer_rank == 2 && src_iter_rank == 2,
                        "Pattern matcher error src_layer, src_iter, have rank 2 for RNN op");
        RETURN_IF_FALSE(src_layer->get_element_type() == element::f32 &&
                            src_iter->get_element_type() == element::f32,
                        "input tensor type and input recurrent state tensor type for RNN op should "
                        "be float32");

        auto rnn = std::make_shared<op::gpu::Rnn>(src_layer,
                                                  src_iter,
                                                  params,
                                                  state_iter,
                                                  num_of_lstm_matched,
                                                  num_gates_in_lstm,
                                                  sequence_len,
                                                  src_layer_feature_size,
                                                  feature_size,
                                                  direction,
                                                  num_fused_rnn_layers);

        std::vector<std::shared_ptr<op::Slice>> ht_slice_per_timestep(num_of_lstm_matched, nullptr);
        auto rnn_ht_out = std::make_shared<op::GetOutputElement>(rnn, 0);
        auto layer_rnn_ht = std::make_shared<op::GetOutputElement>(rnn, 1);
        auto layer_rnn_ct = std::make_shared<op::GetOutputElement>(rnn, 2);

        // slice the rnn ht's
        size_t start_index = 0;
        size_t end_index = batch_size;
        // capture the slices in the reverse order, so it corrosponds to lstm_goes order captured by
        // the Pattern matcher
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

        // we need to collect LSTM from GOE's, in order to determine
        // the individaual time slice output ht. lstm_goes will hold the GOE in the decreasing
        // order of the time slices
        for (size_t i = 0; i < lstm_goes.size(); i++)
        {
            // lstm's will be the input to GOE's
            lstm_nodes.push_back(lstm_goes[i]->get_arguments()[0]);
        }

        RETURN_IF_FALSE(sequence_len == lstm_nodes.size(),
                        " Number of lstm nodes in RNN layer is not equal to time slices");
        RETURN_IF_FALSE(
            lstm_nodes.size() == lstm_goes.size() ||
                lstm_goes.size() == ht_slice_per_timestep.size(),
            "Number of slices of rnn output ht is not equal to the time slices in RNN layer");

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
                    for (auto goe0_user : goe_0->get_users())
                    {
                        if (std::find(lstm_nodes.begin(), lstm_nodes.end(), goe0_user) ==
                                lstm_nodes.end() &&
                            ngraph::is_used(goe0_user.get()))
                        {
                            lstm_goe0_user.insert(goe0_user);
                            map_goe_to_lstm_slices[goe_0] = ht_slice_per_timestep[index];
                            NGRAPH_DEBUG << "ht_slice: " << ht_slice_per_timestep[index]->get_name()
                                         << " goe0_user " << goe0_user->get_name() << " ";
                        }
                    }
                }
                // we need to only check the last LSTM cell Ct user and replace if needed.
                if ((index == 0) && (goe_node->get_n() == 1))
                {
                    // check if the last LSTM cell has any consumers
                    auto n_time_step_lstm_ct_goe = goes->get_node();
                    ngraph::replace_node(n_time_step_lstm_ct_goe, layer_rnn_ct);
                }
            }
        }

        // now go through the lstm goe_0 consumers and replace them with the slice
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
        lstm_node_label, rpattern_ct_1, empty_correlated_matches);
    this->add_matcher(m, callback);
}

static std::shared_ptr<Node>
    compute_multi_layer_rnn_inputs(const std::shared_ptr<pattern::op::Label>& rnn_label,
                                   pattern::RecurrentMatcher& m)
{
    auto node_labels = m.get_bound_nodes_for_pattern(rnn_label);
    std::reverse(node_labels.begin(), node_labels.end());
    return std::make_shared<op::Concat>(node_labels, 0);
}

static std::shared_ptr<Node>
    compute_multi_layer_rnn_params(const std::shared_ptr<pattern::op::Label>& param_label,
                                   pattern::RecurrentMatcher& m)
{
    auto param_nodes = m.get_bound_nodes_for_pattern(param_label);
    std::reverse(param_nodes.begin(), param_nodes.end());

    // iterate over params for each layer in order [layer 0, ... layer n]
    NodeVector biases;
    NodeVector layer_params;
    for (auto& param : param_nodes)
    {
        // split and group layer weights and layer biases
        auto const& args = param->get_arguments();
        for (size_t i = 0; i < args.size(); i++)
        {
            // first half set of params are weights, second half are biases
            if (i < (args.size() / 2))
            {
                layer_params.push_back(args[i]);
            }
            else
            {
                biases.push_back(args[i]);
            }
        }
    }
    layer_params.insert(layer_params.end(), biases.begin(), biases.end());
    return std::make_shared<op::Concat>(layer_params, 0);
}

void ngraph::runtime::gpu::pass::MultiLayerRNNFusion::construct_multi_layer_rnn_fusion_fprop()
{
    auto src_layer_label = std::make_shared<pattern::op::Label>(element::f32, Shape{30, 100});

    auto src_slice =
        std::make_shared<pattern::op::Skip>(src_layer_label, pattern::has_class<op::Slice>());

    auto src_iter_label = std::make_shared<pattern::op::Label>(element::f32, Shape{20, 100});
    auto params_label = std::make_shared<pattern::op::Label>(
        element::f32, Shape{400 * 100 + 400 * 100 + 400 + 400});
    auto state_iter_label = std::make_shared<pattern::op::Label>(element::f32, Shape{20, 100});

    size_t ref_number_of_timesteps = 3;
    size_t ref_number_of_gates_per_cell = 4;
    size_t ref_src_seq_length = 3;
    size_t ref_src_layer_feature_size = 100;
    size_t ref_feature_size = 100;
    size_t ref_rnn_direction = 1;
    size_t ref_num_of_rnn_fused_layer = 1;

    auto ref_rnn_node = std::make_shared<op::gpu::Rnn>(src_slice,
                                                       src_iter_label,
                                                       params_label,
                                                       state_iter_label,
                                                       ref_number_of_timesteps,
                                                       ref_number_of_gates_per_cell,
                                                       ref_src_seq_length,
                                                       ref_src_layer_feature_size,
                                                       ref_feature_size,
                                                       ref_rnn_direction,
                                                       ref_num_of_rnn_fused_layer);

    auto rnn_ht_out = std::make_shared<op::GetOutputElement>(ref_rnn_node, 0);
    auto rnn_ht_label =
        std::make_shared<pattern::op::Label>(rnn_ht_out, nullptr, NodeVector{rnn_ht_out});

    auto callback = [src_layer_label, src_iter_label, params_label, state_iter_label, rnn_ht_label](
        pattern::RecurrentMatcher& m) {

        if (m.get_number_of_recurrent_matches() <= 1)
        {
            return false;
        }

        auto src_nodes = m.get_bound_nodes_for_pattern(src_layer_label);
        auto rnn_ht_out_nodes = m.get_bound_nodes_for_pattern(rnn_ht_label);
        auto number_of_rnn_cell_matched = m.get_number_of_recurrent_matches();
        NGRAPH_DEBUG << "In Recurrent multi layer RNN fusion callback ";
        NGRAPH_DEBUG << "Number of RNN's Matched: " << number_of_rnn_cell_matched;
        NGRAPH_DEBUG << "matched_root: " << m.get_match_root()->get_name();
        NGRAPH_DEBUG << "src_layer_node: " << src_nodes[0]->get_name();

        //  we can fuse across different RNN layers only if SLC == DLC
        for (size_t i = 0; i < number_of_rnn_cell_matched; i++)
        {
            if (src_nodes[i]->get_shape()[1] != rnn_ht_out_nodes[i]->get_shape()[1])
            {
                NGRAPH_DEBUG << "Not fusing since the feature sizes for xt and ht_1 dont match";
                return false;
            }
        }

        // we just need to capture the input symbols {x0 | x1.....| xt} of the first lstm layer
        // the intermediate inputs for the next layer will be computed by the kernel
        auto src_layer_nodes = m.get_bound_nodes_for_pattern(src_layer_label);
        auto src_layer = src_layer_nodes[src_layer_nodes.size() - 1];

        auto src_iter = compute_multi_layer_rnn_inputs(src_iter_label, m);
        auto state_iter = compute_multi_layer_rnn_inputs(state_iter_label, m);
        auto params = compute_multi_layer_rnn_params(params_label, m);

        // collect list of rnn ops (layers)
        std::vector<std::shared_ptr<op::gpu::Rnn>> rnn_nodes;
        for (auto& rnn_goe_input : m.get_bound_nodes_for_pattern(rnn_ht_label))
        {
            auto rnn_op =
                std::dynamic_pointer_cast<op::gpu::Rnn>(rnn_goe_input->get_arguments()[0]);
            if (rnn_op)
            {
                rnn_nodes.push_back(rnn_op);
            }
            else
            {
                throw ngraph_error("Input for RNN output GetOuputElement Op should be RNN");
            }
        }

        size_t num_time_steps = rnn_nodes[0]->get_num_timesteps();
        size_t num_gates_in_lstm = rnn_nodes[0]->get_gates_per_cell();
        size_t batch_size = rnn_nodes[0]->get_batch_size();
        size_t sequence_len = rnn_nodes[0]->get_src_sequence_length();
        size_t src_layer_feature_size = rnn_nodes[0]->get_src_layer_feature_size();
        size_t feature_size = rnn_nodes[0]->get_src_iter_feature_size();
        size_t rnn_direction = rnn_nodes[0]->get_direction();
        size_t num_fused_rnn_layers = m.get_number_of_recurrent_matches();

        NGRAPH_DEBUG << "src_layer: " << join(src_layer->get_shape());
        NGRAPH_DEBUG << "src_iter: " << join(src_iter->get_shape());
        NGRAPH_DEBUG << "state_iter: " << join(state_iter->get_shape());
        NGRAPH_DEBUG << "params size {wx|wh|bx|bh}: " << shape_size(params->get_shape());
        NGRAPH_DEBUG << "src_seq_len: " << sequence_len;
        NGRAPH_DEBUG << "batch_size: " << batch_size;
        NGRAPH_DEBUG << "feature_size: " << feature_size;

        if (auto src_rnn = std::dynamic_pointer_cast<op::gpu::Rnn>(src_layer))
        {
            RETURN_IF_FALSE(
                src_rnn->get_num_timesteps() == num_time_steps,
                "input symbols for the layer fused RNN op, should be captured only for the "
                "first layer");
        }

        RETURN_IF_FALSE(
            !std::dynamic_pointer_cast<op::Parameter>(src_layer) ||
                rnn_nodes[0]->get_num_timesteps() == 1,
            "input symbols for the layer fused RNN op, should be captured only for the first "
            "layer");
        RETURN_IF_FALSE(
            (src_iter->get_arguments().size()) == num_fused_rnn_layers,
            "number of hidden states for RNN op in the layer fusion is not equal to num of "
            "fused_rnn_layers");
        RETURN_IF_FALSE(
            (state_iter->get_arguments().size()) == num_fused_rnn_layers,
            "number of cell states for RNN op in the layer fusion is not equal to num of "
            "fused_rnn_layers");
        RETURN_IF_FALSE(
            (params->get_arguments().size()) == num_fused_rnn_layers * 4,
            "RNN param tensor does not consist of normal and recurrent weight and bias tensor "
            "for each layer");

        auto rnn = std::make_shared<op::gpu::Rnn>(src_layer,
                                                  src_iter,
                                                  params,
                                                  state_iter,
                                                  num_time_steps,
                                                  num_gates_in_lstm,
                                                  sequence_len,
                                                  src_layer_feature_size,
                                                  feature_size,
                                                  rnn_direction,
                                                  num_fused_rnn_layers);

        auto output_layer_rnn_ht = std::make_shared<op::GetOutputElement>(rnn, 0);
        auto layer_rnn_ht = std::make_shared<op::GetOutputElement>(rnn, 1);
        auto layer_rnn_ct = std::make_shared<op::GetOutputElement>(rnn, 2);

        // Replace all the users of RNN cell state {ct} across different user.
        auto replace_rnn_output_cellstate = [&](std::shared_ptr<Node>& rnn_ct, size_t layer) {
            std::shared_ptr<Node> node_to_replace = rnn_ct;
            auto ct_slice = std::make_shared<op::Slice>(
                layer_rnn_ct,
                Coordinate{static_cast<unsigned long>(batch_size * (layer - 1)), 0},
                Coordinate{static_cast<unsigned long>(batch_size * rnn_direction * layer),
                           static_cast<unsigned long>(feature_size)});

            if (rnn_ct->get_users().size() == 1)
            {
                if (std::dynamic_pointer_cast<op::Slice>(rnn_ct->get_users()[0]))
                {
                    node_to_replace = rnn_ct->get_users()[0];
                }
            }
            if (ngraph::is_used(node_to_replace.get()))
            {
                ngraph::replace_node(node_to_replace, ct_slice);
            }
        };

        for (size_t index = 0; index < rnn_nodes.size(); index++)
        {
            for (auto& rnn_goes : rnn_nodes[index]->get_users())
            {
                NGRAPH_DEBUG << "rnn_goes: " << rnn_goes->get_name();
                if (rnn_goes->get_users().empty())
                {
                    continue;
                }

                if (auto rnn_goe_node = std::dynamic_pointer_cast<op::GetOutputElement>(rnn_goes))
                {
                    // we need to only replace the {ht} consumers of the last RNN layer,
                    // since for other layers the intermediate outputs {ht} will be computed
                    // within the kernel
                    if (index == 0)
                    {
                        if (rnn_goe_node->get_n() == 0)
                        {
                            ngraph::replace_node(rnn_goes, output_layer_rnn_ht);
                        }
                    }
                    if (rnn_goe_node->get_n() == 2)
                    {
                        replace_rnn_output_cellstate(rnn_goes, num_fused_rnn_layers - index);
                    }
                }
            }
        }

        return true;
    };

    std::set<std::shared_ptr<pattern::op::Label>> empty_correlated_matches;
    auto m = std::make_shared<pattern::RecurrentMatcher>(
        rnn_ht_label, src_layer_label, empty_correlated_matches);
    this->add_matcher(m, callback);
}
