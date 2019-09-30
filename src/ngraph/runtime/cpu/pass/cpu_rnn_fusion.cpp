//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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
#include "ngraph/op/fused/lstm_cell.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/reverse_sequence.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/op/lstm.hpp"
#include "ngraph/runtime/cpu/op/rnn.hpp"
#include "ngraph/runtime/cpu/op/rnn_utils.hpp"

#define STR(X) #X
#define CHECK_RANK(X, RANK)                                                                        \
    if (X->get_shape().size() != RANK)                                                             \
    {                                                                                              \
        NGRAPH_DEBUG << STR(X) << " does not have rank " << RANK;                                  \
        return false;                                                                              \
    }

using namespace ngraph;

void ngraph::runtime::cpu::pass::LSTMFusion::construct_onnx_lstmcell_fprop()
{
    size_t ref_batch_size = 2;
    size_t ref_input_size = 3;
    size_t ref_hidden_size = 3;
    size_t ref_gates_count = 4;

    auto X =
        std::make_shared<pattern::op::Label>(element::f32, Shape{ref_batch_size, ref_input_size});
    auto W = std::make_shared<pattern::op::Label>(
        element::f32, Shape{ref_gates_count * ref_hidden_size, ref_input_size});
    auto R = std::make_shared<pattern::op::Label>(
        element::f32, Shape{ref_gates_count * ref_hidden_size, ref_hidden_size});
    auto bias_ref = std::make_shared<pattern::op::Label>(
        element::f32, Shape{2 * ref_gates_count * ref_hidden_size});
    auto peep_hole = std::make_shared<pattern::op::Label>(element::f32, Shape{3 * ref_hidden_size});
    auto H_t =
        std::make_shared<pattern::op::Label>(element::f32, Shape{ref_batch_size, ref_hidden_size});
    auto C_t =
        std::make_shared<pattern::op::Label>(element::f32, Shape{ref_batch_size, ref_hidden_size});

    auto ref_lstm_cell =
        std::make_shared<op::LSTMCell>(X,
                                       W,
                                       R,
                                       H_t,
                                       C_t,
                                       ref_hidden_size,
                                       bias_ref,
                                       peep_hole,
                                       std::vector<std::string>{"sigmoid", "tanh", "tanh"},
                                       std::vector<float>{},
                                       std::vector<float>{},
                                       0.f,
                                       false);

    auto callback = [X, W, R, H_t, C_t](pattern::Matcher& m) {

        auto pattern_map = m.get_pattern_map();
        ngraph::runtime::cpu::rnn_utils::rnntype rnn_type =
            ngraph::runtime::cpu::rnn_utils::rnntype::vanilla_lstm;

        auto target_lstm_node = m.get_match_root();
        auto lstmcell_op = std::dynamic_pointer_cast<op::LSTMCell>(m.get_match_root());
        auto src_iter =
            std::make_shared<ngraph::op::Concat>(NodeVector{pattern_map[H_t], pattern_map[C_t]}, 0);
        auto bias_iofc = target_lstm_node->get_argument(5);

        // we need to reorder W, R and bias from IOFC to IFCO gate order
        // Note: ONNX runtime provides W, R and bias in the gate order [IOFC] but
        // MKLDNN computes LSTM kernel in the [IFCO] order.

        auto get_weights_ifco_gate_order =
            [&](std::shared_ptr<Node> weights_graph_node) -> std::shared_ptr<Node> {
            // slices will be in ICFO order
            std::vector<std::shared_ptr<Node>> gate_slices;

            size_t dim0 = weights_graph_node->get_shape()[0] / 4;
            size_t dim1 = weights_graph_node->get_shape()[1];
            for (size_t i = 0; i < 4; i++)
            {
                auto slice = std::make_shared<ngraph::op::Slice>(
                    weights_graph_node, Coordinate{i * dim0, 0}, Coordinate{(i + 1) * dim0, dim1});
                gate_slices.push_back(slice);
            }

            auto weights_ifco = std::make_shared<ngraph::op::Concat>(
                NodeVector{gate_slices[0], gate_slices[2], gate_slices[3], gate_slices[1]}, 0);
            return std::move(weights_ifco);
        };

        auto get_bias_ifco_gate_order =
            [&](std::shared_ptr<Node> bias_graph_node) -> std::shared_ptr<Node> {

            size_t hidden_size = lstmcell_op->get_hidden_size();
            auto Wb_bias = std::make_shared<ngraph::op::Slice>(
                bias_graph_node, Coordinate{0}, Coordinate{4 * hidden_size});
            auto Rb_bias = std::make_shared<ngraph::op::Slice>(
                bias_graph_node, Coordinate{4 * hidden_size}, Coordinate{2 * 4 * hidden_size});
            auto bias = std::make_shared<op::Add>(Wb_bias, Rb_bias);

            // slices will be in ICFO order
            std::vector<std::shared_ptr<Node>> gate_slices;

            for (size_t i = 0; i < 4; i++)
            {
                auto slice = std::make_shared<ngraph::op::Slice>(
                    bias, Coordinate{i * hidden_size}, Coordinate{(i + 1) * hidden_size});
                gate_slices.push_back(slice);
            }

            auto new_bias = std::make_shared<ngraph::op::Concat>(
                NodeVector{gate_slices[0], gate_slices[2], gate_slices[3], gate_slices[1]}, 0);
            return std::move(new_bias);
        };

        auto W_iofc = pattern_map[W];
        auto R_iofc = pattern_map[R];
        auto W_ifco = get_weights_ifco_gate_order(W_iofc);
        auto R_ifco = get_weights_ifco_gate_order(R_iofc);
        // here onnx bias will be of shape (2 * gates_count * hidden_size) bias of Wb and Rb are
        // concatenated, we will split the bias, add and rearrange in order IFCO
        auto bias_ifco = get_bias_ifco_gate_order(bias_iofc);

        auto W_reshape = std::make_shared<op::Reshape>(
            W_ifco, AxisVector{1, 0}, Shape{W_ifco->get_shape()[1], W_ifco->get_shape()[0]});
        auto R_reshape = std::make_shared<op::Reshape>(
            R_ifco, AxisVector{1, 0}, Shape{R_ifco->get_shape()[1], R_ifco->get_shape()[0]});

#if MKLDNN_VERSION_MAJOR < 1
        auto lstm_node = std::make_shared<ngraph::op::Lstm>(
            pattern_map[X], src_iter, W_reshape, R_reshape, bias_ifco, rnn_type);

        if (lstm_node->get_outputs().size() != 2)
        {
            throw ngraph_error("Lstm node doesnt have two outputs");
        }
#else
        auto lstm_node = std::make_shared<ngraph::op::Lstm>(pattern_map[X],
                                                            pattern_map[H_t],
                                                            pattern_map[C_t],
                                                            W_reshape,
                                                            R_reshape,
                                                            bias_ifco,
                                                            rnn_type);
        if (lstm_node->get_outputs().size() != 3)
        {
            throw ngraph_error("Lstm node doesnt have three outputs");
        }
#endif

#if MKLDNN_VERSION_MAJOR < 1
        auto lstm_ht_output = std::make_shared<ngraph::op::GetOutputElement>(lstm_node, 0);
        auto lstm_ht_ct_output = std::make_shared<ngraph::op::GetOutputElement>(lstm_node, 1);
#else
        auto lstm_ht_output = std::make_shared<ngraph::op::GetOutputElement>(lstm_node, 1);
        auto ct_slice = std::make_shared<ngraph::op::GetOutputElement>(lstm_node, 2);
#endif

        auto goe_nodes = ngraph::op::get_output_elements(m.get_match_root());
        auto dst_layer = goe_nodes[0];
        auto dst_iter = goe_nodes[1];
// dst_iter of lstm mkldnn output holds the results of both recurrent state
// tensor outputs. we need to slice the ct.
#if MKLDNN_VERSION_MAJOR < 1
        // set LSTM cell attributes
        const size_t lstm_n_gates = 4;
        const size_t direction = 1;
        const size_t layers = 1;
        const size_t batch_size = pattern_map[X]->get_shape()[0];
        auto dic = pattern_map[R]->get_shape()[0] / (lstm_n_gates * direction * layers);
        auto ct_slice = std::make_shared<ngraph::op::Slice>(
            lstm_ht_ct_output, Coordinate{batch_size, 0}, Coordinate{(2 * batch_size), dic});
#endif
        // find the user's for {ht} and replace them with lstm_goe_0
        if (std::dynamic_pointer_cast<ngraph::op::GetOutputElement>(dst_iter) != nullptr)
        {
            ngraph::replace_node(dst_iter, ct_slice);
        }
        // find the user's for {ht} and replace them with lstm_goe_0
        if (std::dynamic_pointer_cast<ngraph::op::GetOutputElement>(dst_layer) != nullptr)
        {
            ngraph::replace_node(dst_layer, lstm_ht_output);
        }
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(ref_lstm_cell, "LSTMFusion.onnx_lstm_cell");
    this->add_matcher(m, callback);
}

void ngraph::runtime::cpu::pass::LSTMFusion::construct_sigmoid()
{
    // construct variance
    auto input = std::make_shared<pattern::op::Label>(element::f32, Shape{3, 4});
    auto neg_input = std::make_shared<ngraph::op::Negative>(input);
    auto exp_neg_input = std::make_shared<ngraph::op::Exp>(neg_input);

    // broadcast input
    auto constant = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto broadcast_constant =
        std::make_shared<ngraph::op::Broadcast>(constant, Shape{3, 4}, AxisSet{0, 1});

    auto add_exp = std::make_shared<ngraph::op::Add>(exp_neg_input, broadcast_constant);
    auto divide_1_over_exp = std::make_shared<ngraph::op::Divide>(broadcast_constant, add_exp);

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

        auto sigmoid_node = std::make_shared<ngraph::op::Sigmoid>(pattern_map[input]);
        ngraph::replace_node(m.get_match_root(), sigmoid_node);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(divide_1_over_exp, "LSTMFusion.Sigmoid");
    this->add_matcher(m, callback);
}

static void replace_collapse_node_user(std::shared_ptr<Node> collapsed_node,
                                       const Output<Node>& new_output)
{
    for (auto node : collapsed_node->get_users(true))
    {
        NGRAPH_DEBUG << "node_name: " << node->get_name();
        for (size_t i = 0; i < node->get_input_size(); i++)
        {
            if (node->input(i).get_source_output().get_node_shared_ptr() == collapsed_node)
            {
                node->set_argument(i, new_output);
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
        return ((std::dynamic_pointer_cast<ngraph::op::Broadcast>(n) != nullptr) ||
                (std::dynamic_pointer_cast<ngraph::op::Reshape>(n) != nullptr));
    };

    // Fused MatMuls
    // (W_{ii} | (W_{if} | W_{ig} | W_{io}) * x_t + (b_{ii} | b_{if} |  b_{ig} | b_{io})
    auto dot1 = std::make_shared<ngraph::op::Dot>(xt, w_i2h);
    auto add1 = std::make_shared<ngraph::op::Add>(
        dot1, std::make_shared<pattern::op::Skip>(bias_i2h, broadcast_pred));
    // (W_{hi} | (W_{hf} | W_{hg} | W_{ho}) * h_{(t-1)} + (b_{hi} | b_{hf} |  b_{hg} | b_{ho})
    auto dot2 = std::make_shared<ngraph::op::Dot>(ht_1, w_h2h);
    auto add2 = std::make_shared<ngraph::op::Add>(
        dot2, std::make_shared<pattern::op::Skip>(bias_h2h, broadcast_pred));

    auto X = std::make_shared<ngraph::op::Add>(add2, add1);

    // construct gates
    auto it = std::make_shared<ngraph::op::Sigmoid>(
        std::make_shared<ngraph::op::Slice>(X, Coordinate{0, 0}, Coordinate{10, 100}));
    auto ft = std::make_shared<ngraph::op::Sigmoid>(
        std::make_shared<ngraph::op::Slice>(X, Coordinate{0, 100}, Coordinate{10, 200}));
    auto gt = std::make_shared<ngraph::op::Tanh>(
        std::make_shared<ngraph::op::Slice>(X, Coordinate{0, 200}, Coordinate{10, 300}));
    auto ot = std::make_shared<ngraph::op::Sigmoid>(
        std::make_shared<ngraph::op::Slice>(X, Coordinate{0, 300}, Coordinate{10, 400}));

    // construct (c_t) cell state
    auto ct = std::make_shared<ngraph::op::Add>(std::make_shared<ngraph::op::Multiply>(ft, ct_1),
                                                std::make_shared<ngraph::op::Multiply>(it, gt));
    auto ct_label = std::make_shared<pattern::op::Label>(ct, nullptr, NodeVector{ct});

    // construct (h_t)
    auto ht =
        std::make_shared<ngraph::op::Multiply>(ot, std::make_shared<ngraph::op::Tanh>(ct_label));

    // Define a call back that needs to called once the DFG matches the pattern
    auto callback = [ct_label, w_i2h, bias_i2h, w_h2h, bias_h2h, xt, ht_1, ct_1](
        pattern::Matcher& m) {
        NGRAPH_DEBUG << "In a callback for construct_fprop_lstm pattern against "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        if (m.get_match_root()->get_element_type() != element::f32)
        {
            NGRAPH_DEBUG << "mpattern = " << m.get_match_root()->get_name()
                         << " type is not float!";
            return false;
        }

        CHECK_RANK(pattern_map[xt], 2)
        CHECK_RANK(pattern_map[ht_1], 2)
        CHECK_RANK(pattern_map[w_i2h], 2)
        CHECK_RANK(pattern_map[w_h2h], 2)
        CHECK_RANK(pattern_map[bias_i2h], 1)
        CHECK_RANK(pattern_map[bias_h2h], 1)

        auto weights_layer = pattern_map[w_i2h];
        auto weights_iter = pattern_map[w_h2h];
        auto src_layer = pattern_map[xt];
        auto hidden_state = pattern_map[ht_1];
        auto cell_state = pattern_map[ct_1];

// TODO: (Pruthvi) temporary workaround for GNMT slow down
// this checks avoids fusing of LSTM cells if its a part of decoder, we
// will remove this once mkldnn optimizes individual LSTM cell or once
// we have decoder pattern for GNMT.
#if MKLDNN_VERSION_MAJOR < 1
        if (!(std::dynamic_pointer_cast<ngraph::op::Broadcast>(cell_state) &&
              std::dynamic_pointer_cast<ngraph::op::Constant>(cell_state->get_argument(0))) &&
            !(std::dynamic_pointer_cast<ngraph::op::Slice>(cell_state) &&
              std::dynamic_pointer_cast<ngraph::op::GetOutputElement>(cell_state->get_argument(0))))
        {
            return false;
        }
#else
        if (!(std::dynamic_pointer_cast<ngraph::op::Broadcast>(cell_state) &&
              std::dynamic_pointer_cast<ngraph::op::Constant>(cell_state->get_argument(0))) &&
            !(std::dynamic_pointer_cast<ngraph::op::GetOutputElement>(cell_state)))
        {
            return false;
        }
#endif

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
        if (std::dynamic_pointer_cast<ngraph::op::Broadcast>(src_layer) &&
            std::dynamic_pointer_cast<ngraph::op::Constant>(src_layer->get_argument(0)))
        {
            // First timestep of an RNN layer
            swap_lstm_inputs();
        }
        else if (hidden_state->get_shape() != cell_state->get_shape())
        {
            swap_lstm_inputs();
        }
#if MKLDNN_VERSION_MAJOR < 1
        else if (std::dynamic_pointer_cast<ngraph::op::GetOutputElement>(
                     cell_state->get_argument(0)))
        {
            // swap the inputs if the cell_state and hidden state does not
            // belong to the same Lstm
            if (!hidden_state->get_argument(0)->get_arguments().size() ||
                (hidden_state->get_argument(0)->get_arguments()[0] !=
                 cell_state->get_argument(0)->get_arguments()[0]))
            {
                swap_lstm_inputs();
            }
        }
#else
        else if (std::dynamic_pointer_cast<ngraph::op::GetOutputElement>(cell_state))
        {
            // swap the inputs if the cell_state and hidden state does not
            // belong to the same Lstm
            if (hidden_state->input(0).get_source_output().get_node() !=
                cell_state->input(0).get_source_output().get_node())
            {
                swap_lstm_inputs();
            }
        }
#endif

        if (hidden_state->get_shape() != cell_state->get_shape())
        {
            NGRAPH_DEBUG << "Lstm MKLDNN kernel requires recurrent output hidden states to match ";
            return false;
        }

        // set LSTM cell attributes
        size_t lstm_n_gates = 4;
        size_t direction = 1;
        size_t layers = 1;
        auto dlc = weights_layer->get_shape()[1] / (lstm_n_gates * direction * layers);
        auto slc = weights_layer->get_shape()[0];
        auto dic = weights_iter->get_shape()[1] / (lstm_n_gates * direction * layers);
        auto sic = weights_iter->get_shape()[0];
        ngraph::runtime::cpu::rnn_utils::rnntype rnn_type =
            ngraph::runtime::cpu::rnn_utils::rnntype::vanilla_lstm;

        if (dlc != dic)
        {
            NGRAPH_DEBUG << "Not fusing, since Lstm kernel requires dst_layer feature size "
                         << "equals to dts_iter feature size";
            return false;
        }

        auto bias = std::make_shared<ngraph::op::Add>(pattern_map[bias_i2h], pattern_map[bias_h2h]);

#if MKLDNN_VERSION_MAJOR < 1
        size_t batch_size = src_layer->get_shape()[0];
        std::shared_ptr<Node> src_iter =
            std::make_shared<ngraph::op::Concat>(NodeVector{hidden_state, cell_state}, 0);
        if (src_layer->get_shape()[1] != slc || src_iter->get_shape()[1] != sic)
        {
            NGRAPH_DEBUG << "Feature size mismatch between weights and input tensors";
            return false;
        }

        auto lstm_node = std::make_shared<ngraph::op::Lstm>(
            src_layer, src_iter, weights_layer, weights_iter, bias, rnn_type);

        auto lstm_ht_output = std::make_shared<ngraph::op::GetOutputElement>(lstm_node, 0);
        auto lstm_ht_ct_output = std::make_shared<ngraph::op::GetOutputElement>(lstm_node, 1);

        // dst_iter of lstm mkldnn output holds the results of both recurrent state
        // tensor outputs. we need to slice the ct.
        auto ht_slice = std::make_shared<ngraph::op::Slice>(
            lstm_ht_output, Coordinate{0, 0}, Coordinate{batch_size, dlc});
        auto ct_slice = std::make_shared<ngraph::op::Slice>(
            lstm_ht_ct_output, Coordinate{batch_size, 0}, Coordinate{(2 * batch_size), dic});

        // Now identify the nodes which consumes the output of LSTM nodes
        // and replace them accordingly
        // find the user's for {ht|ct} and replace them with lstm_goe_1
        if (ngraph::is_used(pattern_map[ct_label].get()))
        {
            replace_collapse_node_user(pattern_map[ct_label], ct_slice->output(0));
        }
        // find the user's for {ht} and replace them with lstm_goe_0
        ngraph::replace_node(m.get_match_root(), ht_slice);
#else
        if (src_layer->get_shape()[1] != slc || hidden_state->get_shape()[1] != sic ||
            cell_state->get_shape()[1] != sic)
        {
            NGRAPH_DEBUG << "Feature size mismatch between weights and input tensors";
            return false;
        }
        auto lstm_node = std::make_shared<ngraph::op::Lstm>(
            src_layer, hidden_state, cell_state, weights_layer, weights_iter, bias, rnn_type);

        auto lstm_ht_output = std::make_shared<ngraph::op::GetOutputElement>(lstm_node, 1);
        auto lstm_ct_output = std::make_shared<ngraph::op::GetOutputElement>(lstm_node, 2);

        // Now identify the nodes which consumes the output of LSTM nodes
        // and replace them accordingly
        // find the user's for {ct} and replace them with lstm_goe_2
        if (ngraph::is_used(pattern_map[ct_label].get()))
        {
            replace_collapse_node_user(pattern_map[ct_label], lstm_ct_output->output(0));
        }
        // find the user's for {ht} and replace them with lstm_goe_1
        ngraph::replace_node(m.get_match_root(), lstm_ht_output);
#endif
        return true;
    };
    auto m = std::make_shared<pattern::Matcher>(ht, "LSTMFusion.Fprop");
    this->add_matcher(m, callback);
}

void ngraph::runtime::cpu::pass::RNNFusion::construct_rnn_lstm_fprop()
{
    // Captures multiple LSTM cells corresponding to the timesteps of a single RNN
    // Shared nodes across LSTM cells -
    auto lstm_src_layer = std::make_shared<pattern::op::Label>(element::f32, Shape{30, 100});

    auto lstm_ht = std::make_shared<pattern::op::Label>(element::f32, Shape{10, 100});
    auto lstm_ct = std::make_shared<pattern::op::Label>(element::f32, Shape{10, 100});

#if MKLDNN_VERSION_MAJOR < 1
    auto lstm_src_iter = std::make_shared<ngraph::op::Concat>(NodeVector{lstm_ht, lstm_ct}, 0);
    auto lstm_src_iter_label =
        std::make_shared<pattern::op::Label>(lstm_src_iter, nullptr, NodeVector{lstm_src_iter});
#endif

    auto lstm_weights_layer_shared = std::make_shared<pattern::op::Label>(
        element::f32, Shape{400, 100}, pattern::has_class<ngraph::op::Parameter>());
    auto lstm_weights_layer = std::make_shared<ngraph::op::Reshape>(
        lstm_weights_layer_shared, AxisVector{1, 0}, Shape{100, 400});
    auto lstm_weights_layer_label = std::make_shared<pattern::op::Label>(
        lstm_weights_layer, nullptr, NodeVector{lstm_weights_layer});

    auto lstm_weights_iter_shared = std::make_shared<pattern::op::Label>(
        element::f32, Shape{400, 100}, pattern::has_class<ngraph::op::Parameter>());
    auto lstm_weights_iter = std::make_shared<ngraph::op::Reshape>(
        lstm_weights_iter_shared, AxisVector{1, 0}, Shape{100, 400});
    auto lstm_weights_iter_label = std::make_shared<pattern::op::Label>(
        lstm_weights_iter, nullptr, NodeVector{lstm_weights_iter});

    auto lstm_bias_layer_shared = std::make_shared<pattern::op::Label>(element::f32, Shape{400});
    auto lstm_bias_iter_shared = std::make_shared<pattern::op::Label>(element::f32, Shape{400});
    auto lstm_bias =
        std::make_shared<ngraph::op::Add>(lstm_bias_layer_shared, lstm_bias_iter_shared);
    auto lstm_bias_label =
        std::make_shared<pattern::op::Label>(lstm_bias, nullptr, NodeVector{lstm_bias});
    ngraph::runtime::cpu::rnn_utils::rnntype ref_rnn_type =
        ngraph::runtime::cpu::rnn_utils::rnntype::vanilla_lstm;

#if MKLDNN_VERSION_MAJOR < 1
    auto lstm = std::make_shared<ngraph::op::Lstm>(lstm_src_layer,
                                                   lstm_src_iter_label,
                                                   lstm_weights_layer_label,
                                                   lstm_weights_iter_label,
                                                   lstm_bias_label,
                                                   ref_rnn_type);
    auto lstm_goe = std::make_shared<ngraph::op::GetOutputElement>(lstm, 1);
    // We cannot attach labels to multi-output nodes, so we attach a label to the goe instead
    auto lstm_goe_label =
        std::make_shared<pattern::op::Label>(lstm_goe, nullptr, NodeVector{lstm_goe});
    auto lstm_goe_slice =
        std::make_shared<ngraph::op::Slice>(lstm_goe_label, Coordinate{10, 0}, Coordinate{20, 100});

    auto callback = [lstm_goe_label,
                     lstm_src_layer,
                     lstm_src_iter_label,
                     lstm_weights_layer_label,
                     lstm_weights_iter_label,
                     lstm_bias_label](pattern::RecurrentMatcher& m) {

        NGRAPH_DEBUG << " In recurrent RNN fusion callback";

        auto concat_rnn_inputs_across_timestep =
            [&](std::shared_ptr<pattern::op::Label> input_label) -> std::shared_ptr<Node> {
            NodeVector concat_args;
            // src_layer -> concatenate input symbols from different LSTM cells belonging to same
            // RNN layer
            // in the order 0, 1, 2... t time slice
            {
                auto node_labels = m.get_bound_nodes_for_pattern(input_label);
                std::reverse(node_labels.begin(), node_labels.end());
                return std::make_shared<ngraph::op::Concat>(node_labels, 0);
            }
        };

        const auto sequence_len = m.get_number_of_recurrent_matches();
        if (sequence_len < 2)
        {
            NGRAPH_DEBUG << "Single timestep RNN";
            return false;
        }

        auto rnn_src_layer = concat_rnn_inputs_across_timestep(lstm_src_layer);
        // pick src_iter from first lstm
        auto rnn_src_iter = m.get_bound_nodes_for_pattern(lstm_src_iter_label)[sequence_len - 1];
        // weights and bias are shared across lstms. so pick any
        auto rnn_weights_layer = m.get_bound_nodes_for_pattern(lstm_weights_layer_label)[0];
        auto rnn_weights_iter = m.get_bound_nodes_for_pattern(lstm_weights_iter_label)[0];
        auto rnn_bias = m.get_bound_nodes_for_pattern(lstm_bias_label)[0];

        const size_t lstm_n_gates = 4;
        const size_t batch_size = rnn_src_layer->get_shape()[0] / sequence_len;
        const size_t src_iter_feature_size = rnn_weights_iter->get_shape()[0];
        const size_t num_cell_states = 2;
        const size_t direction = 1;
        const size_t num_fused_rnn_layers = 1;
        ngraph::runtime::cpu::rnn_utils::rnntype rnn_type =
            ngraph::runtime::cpu::rnn_utils::rnntype::vanilla_lstm;

        NGRAPH_DEBUG << "src_layer: " << join(rnn_src_layer->get_shape());
        NGRAPH_DEBUG << "src_iter: " << join(rnn_src_iter->get_shape());
        NGRAPH_DEBUG << "weights_layer: " << join(rnn_weights_layer->get_shape());
        NGRAPH_DEBUG << "weights_iter: " << join(rnn_weights_iter->get_shape());
        NGRAPH_DEBUG << "bias: " << join(rnn_bias->get_shape());
        NGRAPH_DEBUG << "src_seq_len: " << sequence_len;
        NGRAPH_DEBUG << "batch_size: " << batch_size;

        if ((rnn_src_iter->get_arguments().size()) != num_cell_states)
        {
            throw ngraph_error("number of states for RNN op is not equal to (ht_1|ct_1)");
        }

        auto check_const_input = [&](std::shared_ptr<Node> n) {
            if (std::dynamic_pointer_cast<ngraph::op::Constant>(n) ||
                (std::dynamic_pointer_cast<ngraph::op::Broadcast>(n) &&
                 std::dynamic_pointer_cast<ngraph::op::Constant>(n->get_argument(0))))
            {
                return true;
            }
            return false;
        };

        for (size_t i = 0; i < num_cell_states; i++)
        {
            if (!check_const_input(rnn_src_iter->get_argument(i)))
            {
                NGRAPH_DEBUG << "Non const input for RNN state initializer";
                return false;
            }
        }

        CHECK_RANK(rnn_src_layer, 2)
        CHECK_RANK(rnn_src_iter, 2)
        CHECK_RANK(rnn_weights_layer, 2)
        CHECK_RANK(rnn_weights_iter, 2)
        CHECK_RANK(rnn_bias, 1)

        if (rnn_src_layer->get_element_type() != element::f32 ||
            rnn_src_iter->get_element_type() != element::f32)
        {
            NGRAPH_DEBUG << "input tensor type and input recurrent state tensor are not float32";
            return false;
        }

        auto rnn = std::make_shared<ngraph::op::Rnn>(rnn_src_layer,
                                                     rnn_src_iter,
                                                     rnn_weights_layer,
                                                     rnn_weights_iter,
                                                     rnn_bias,
                                                     sequence_len,
                                                     lstm_n_gates,
                                                     sequence_len,
                                                     num_cell_states,
                                                     direction,
                                                     num_fused_rnn_layers,
                                                     rnn_type);

        std::vector<std::shared_ptr<ngraph::op::Slice>> ht_slice_per_timestep(sequence_len,
                                                                              nullptr);
        auto rnn_ht_goe = std::make_shared<ngraph::op::GetOutputElement>(rnn, 0);
        auto rnn_ht_ct_goe = std::make_shared<ngraph::op::GetOutputElement>(rnn, 1);

        for (size_t i = 0, start_index = 0; i < sequence_len; i++, start_index += batch_size)
        {
            ht_slice_per_timestep[i] = (std::make_shared<ngraph::op::Slice>(
                rnn_ht_goe,
                Coordinate{start_index, 0},
                Coordinate{start_index + batch_size, src_iter_feature_size}));
        }

        // find the lstm's nodes captured in PM
        auto lstm_goes = m.get_bound_nodes_for_pattern(lstm_goe_label);
        std::reverse(lstm_goes.begin(), lstm_goes.end());
        std::vector<std::shared_ptr<ngraph::Node>> lstm_nodes;

        // we need to collect LSTM from GOE's, in order to deterministicaly determine
        // the individaual time slice output ht.
        for (size_t i = 0; i < sequence_len; i++)
        {
            // lstm's will be the input to GOE's
            lstm_nodes.push_back(lstm_goes[i]->get_arguments()[0]);
        }

        // collect all the consumers of LSTM goe's (ht)
        std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node>> map_to_rnn_slices;

        for (size_t index = 0; index < sequence_len; index++)
        {
            auto goe_nodes = ngraph::op::get_output_elements(lstm_nodes[index]);

            // if there is no GOE followed by the Lstm, their might be pattern match error
            // we will return safely
            if (goe_nodes.size() != 2)
            {
                return false;
            }

            // dst_layer of the lstm cell
            auto goe_0 = goe_nodes[0];

            if (goe_0)
            {
                for (auto goe0_user : goe_0->get_users())
                {
                    if (ngraph::is_used(goe0_user.get()))
                    {
                        if (!std::dynamic_pointer_cast<ngraph::op::Slice>(goe0_user))
                        {
                            NGRAPH_DEBUG << "Did not find LSTM slice to replace with RNN slice";
                            return false;
                        }
                        map_to_rnn_slices.insert(
                            make_pair(goe0_user, ht_slice_per_timestep[index]));

                        NGRAPH_DEBUG << "ht_slice: " << ht_slice_per_timestep[index]->get_name()
                                     << " goe0_user " << goe0_user->get_name() << " ";
                    }
                }
            }
        }

        auto rnn_ct_goe = ngraph::op::get_output_elements(lstm_nodes[sequence_len - 1])[1];
        if (rnn_ct_goe)
        {
            replace_collapse_node_user(rnn_ct_goe, rnn_ht_ct_goe->output(0));
        }

        // now go through the lstm goe_0 consumers and replace them with the slice
        for (auto& a : map_to_rnn_slices)
        {
            ngraph::replace_node(a.first, a.second);
        }
        NGRAPH_DEBUG << "End of recurrent fusion call back "
                     << "matched_node: " << m.get_match_root()->get_name();
        return true;
    };

    auto m = std::make_shared<pattern::RecurrentMatcher>(
        lstm_goe_slice,
        lstm_ct,
        std::set<std::shared_ptr<pattern::op::Label>>{lstm_weights_layer_shared,
                                                      lstm_weights_iter_shared,
                                                      lstm_bias_layer_shared,
                                                      lstm_bias_iter_shared});
#else
    auto lstm = std::make_shared<ngraph::op::Lstm>(lstm_src_layer,
                                                   lstm_ht,
                                                   lstm_ct,
                                                   lstm_weights_layer_label,
                                                   lstm_weights_iter_label,
                                                   lstm_bias_label,
                                                   ref_rnn_type);
    auto lstm_goe = std::make_shared<ngraph::op::GetOutputElement>(lstm, 2);
    // We cannot attach labels to multi-output nodes, so we attach a label to the goe instead
    auto lstm_goe_label =
        std::make_shared<pattern::op::Label>(lstm_goe, nullptr, NodeVector{lstm_goe});

    auto callback = [lstm_goe_label,
                     lstm_src_layer,
                     lstm_ht,
                     lstm_ct,
                     lstm_weights_layer_label,
                     lstm_weights_iter_label,
                     lstm_bias_label](pattern::RecurrentMatcher& m) {

        NGRAPH_DEBUG << " In recurrent RNN fusion callback";

        auto concat_rnn_inputs_across_timestep =
            [&](std::shared_ptr<pattern::op::Label> input_label) -> std::shared_ptr<Node> {
            NodeVector concat_args;
            // src_layer -> concatenate input symbols from different LSTM cells belonging to same
            // RNN layer
            // in the order 0, 1, 2... t time slice
            {
                auto node_labels = m.get_bound_nodes_for_pattern(input_label);
                std::reverse(node_labels.begin(), node_labels.end());
                return std::make_shared<ngraph::op::Concat>(node_labels, 0);
            }
        };

        const auto sequence_len = m.get_number_of_recurrent_matches();
        if (sequence_len < 2)
        {
            NGRAPH_DEBUG << "Single timestep RNN";
            return false;
        }

        auto rnn_src_layer = concat_rnn_inputs_across_timestep(lstm_src_layer);
        // pick src_iter from first lstm
        auto rnn_src_iter = m.get_bound_nodes_for_pattern(lstm_ht)[sequence_len - 1];
        // pick src_iter_c from first lstm
        auto rnn_src_iter_c = m.get_bound_nodes_for_pattern(lstm_ct)[sequence_len - 1];
        // weights and bias are shared across lstms. so pick any
        auto rnn_weights_layer = m.get_bound_nodes_for_pattern(lstm_weights_layer_label)[0];
        auto rnn_weights_iter = m.get_bound_nodes_for_pattern(lstm_weights_iter_label)[0];
        auto rnn_bias = m.get_bound_nodes_for_pattern(lstm_bias_label)[0];

        const size_t lstm_n_gates = 4;
        const size_t batch_size = rnn_src_layer->get_shape()[0] / sequence_len;
        const size_t src_iter_feature_size = rnn_weights_iter->get_shape()[0];
        const size_t num_cell_states = 2;
        const size_t direction = 1;
        const size_t num_fused_rnn_layers = 1;
        ngraph::runtime::cpu::rnn_utils::rnntype rnn_type =
            ngraph::runtime::cpu::rnn_utils::rnntype::vanilla_lstm;

        NGRAPH_DEBUG << "src_layer: " << join(rnn_src_layer->get_shape());
        NGRAPH_DEBUG << "src_iter: " << join(rnn_src_iter->get_shape());
        NGRAPH_DEBUG << "src_iter_c: " << join(rnn_src_iter_c->get_shape());
        NGRAPH_DEBUG << "weights_layer: " << join(rnn_weights_layer->get_shape());
        NGRAPH_DEBUG << "weights_iter: " << join(rnn_weights_iter->get_shape());
        NGRAPH_DEBUG << "bias: " << join(rnn_bias->get_shape());
        NGRAPH_DEBUG << "src_seq_len: " << sequence_len;
        NGRAPH_DEBUG << "batch_size: " << batch_size;

        auto check_const_input = [&](std::shared_ptr<Node> n) {
            if (std::dynamic_pointer_cast<ngraph::op::Constant>(n) ||
                (std::dynamic_pointer_cast<ngraph::op::Broadcast>(n) &&
                 std::dynamic_pointer_cast<ngraph::op::Constant>(n->get_argument(0))))
            {
                return true;
            }
            return false;
        };

        if (!check_const_input(rnn_src_iter->get_argument(0)) ||
            !check_const_input(rnn_src_iter_c->get_argument(0)))
        {
            NGRAPH_DEBUG << "Non const input for RNN state initializer";
            return false;
        }

        CHECK_RANK(rnn_src_layer, 2);
        CHECK_RANK(rnn_src_iter, 2);
        CHECK_RANK(rnn_src_iter_c, 2);
        CHECK_RANK(rnn_weights_layer, 2);
        CHECK_RANK(rnn_weights_iter, 2);
        CHECK_RANK(rnn_bias, 1);

        if (rnn_src_layer->get_element_type() != element::f32 ||
            rnn_src_iter->get_element_type() != element::f32 ||
            rnn_src_iter_c->get_element_type() != element::f32)
        {
            NGRAPH_DEBUG << "input tensor type and input recurrent state tensor are not float32";
            return false;
        }

        auto rnn = std::make_shared<ngraph::op::Rnn>(rnn_src_layer,
                                                     rnn_src_iter,
                                                     rnn_src_iter_c,
                                                     rnn_weights_layer,
                                                     rnn_weights_iter,
                                                     rnn_bias,
                                                     sequence_len,
                                                     lstm_n_gates,
                                                     sequence_len,
                                                     num_cell_states,
                                                     direction,
                                                     num_fused_rnn_layers,
                                                     rnn_type);

        std::vector<std::shared_ptr<ngraph::op::Slice>> ht_slice_per_timestep(sequence_len,
                                                                              nullptr);
        auto rnn_hts_goe = std::make_shared<ngraph::op::GetOutputElement>(rnn, 0);
        auto rnn_ct_goe = std::make_shared<ngraph::op::GetOutputElement>(rnn, 2);

        for (size_t i = 0, start_index = 0; i < sequence_len; i++, start_index += batch_size)
        {
            ht_slice_per_timestep[i] = (std::make_shared<ngraph::op::Slice>(
                rnn_hts_goe,
                Coordinate{start_index, 0},
                Coordinate{start_index + batch_size, src_iter_feature_size}));
        }

        // find the lstm's nodes captured in PM
        auto lstm_cts = m.get_bound_nodes_for_pattern(lstm_ct);
        std::reverse(lstm_cts.begin(), lstm_cts.end());
        std::vector<std::shared_ptr<ngraph::Node>> lstm_nodes;

        // we need to collect LSTM from GOE's, in order to deterministically determine
        // the individual time slice output ht.
        for (size_t i = 0; i < sequence_len; i++)
        {
            // lstm's will be the user of lstm_ct
            for (auto user : lstm_cts[i]->get_users())
            {
                if (std::dynamic_pointer_cast<ngraph::op::Lstm>(user))
                {
                    lstm_nodes.push_back(user);
                    break;
                }
            }
        }

        // replace LSTM dst_iter with RNN dst_layer slice for LSTM dst_iter users (not including the
        // LSTM in the same layer)
        for (size_t index = 0; index < sequence_len; index++)
        {
            auto goe_nodes = ngraph::op::get_output_elements(lstm_nodes[index]);

            // if there is no GOE followed by the Lstm, their might be pattern match error
            // we will return safely
            if (goe_nodes.size() != 3)
            {
                return false;
            }

            // dst_iter of the lstm cell
            auto goe_1 = goe_nodes[1];
            if (goe_1)
            {
                for (auto goe1_user : goe_1->get_users())
                {
                    // do not include LSTM in the same layer
                    if (std::find(lstm_nodes.begin(), lstm_nodes.end(), goe1_user) ==
                        lstm_nodes.end())
                    {
                        for (size_t i = 0; i < goe1_user->get_input_size(); i++)
                        {
                            if (goe1_user->get_argument(i) == goe_1)
                            {
                                goe1_user->get_inputs().at(i).replace_output(
                                    ht_slice_per_timestep[index]->get_outputs().at(0));
                            }
                        }
                        NGRAPH_DEBUG << "ht_slice: " << ht_slice_per_timestep[index]->get_name()
                                     << " goe1_user " << goe1_user->get_name() << " ";
                    }
                }
            }
        }

        // replace last LSTM dst_iter_c with RNN dst iter_c for last LSTM dst_iter_c's users
        auto last_lstm_ct_goe = ngraph::op::get_output_elements(lstm_nodes[sequence_len - 1])[2];
        if (last_lstm_ct_goe)
        {
            replace_collapse_node_user(last_lstm_ct_goe, rnn_ct_goe->output(0));
        }

        NGRAPH_DEBUG << "End of recurrent fusion call back "
                     << "matched_node: " << m.get_match_root()->get_name();
        return true;
    };

    auto m = std::make_shared<pattern::RecurrentMatcher>(
        lstm_goe,
        lstm_ct,
        std::set<std::shared_ptr<pattern::op::Label>>{lstm_weights_layer_shared,
                                                      lstm_weights_iter_shared,
                                                      lstm_bias_layer_shared,
                                                      lstm_bias_iter_shared});
#endif
    this->add_matcher(m, callback);
}

static std::shared_ptr<Node> stack_rnn_inputs(NodeVector rnn_input_nodes)
{
    std::reverse(rnn_input_nodes.begin(), rnn_input_nodes.end());
    return std::make_shared<ngraph::op::Concat>(rnn_input_nodes, 0);
}

void ngraph::runtime::cpu::pass::MultiLayerRNNFusion::construct_multi_layer_rnn_fusion_fprop()
{
    auto rnn_src_layer = std::make_shared<pattern::op::Label>(element::f32, Shape{30, 100});
#if MKLDNN_VERSION_MAJOR < 1
    auto rnn_src_iter = std::make_shared<pattern::op::Label>(element::f32, Shape{20, 100});
#else
    auto rnn_src_iter = std::make_shared<pattern::op::Label>(element::f32, Shape{10, 100});
    auto rnn_src_iter_c = std::make_shared<pattern::op::Label>(element::f32, Shape{10, 100});
#endif
    auto rnn_weights_layer = std::make_shared<pattern::op::Label>(element::f32, Shape{100, 400});
    auto rnn_weights_iter = std::make_shared<pattern::op::Label>(element::f32, Shape{100, 400});
    auto rnn_bias = std::make_shared<pattern::op::Label>(element::f32, Shape{400});
    const size_t ref_number_of_timesteps = 3;
    const size_t ref_number_of_gates_per_cell = 4;
    const size_t ref_src_seq_length = 3;
    const size_t ref_num_rnn_cell_states = 2;
    const size_t ref_rnn_direction = 1;
    const size_t ref_num_of_rnn_fused_layer = 1;
    ngraph::runtime::cpu::rnn_utils::rnntype ref_rnn_type =
        ngraph::runtime::cpu::rnn_utils::rnntype::vanilla_lstm;

    auto ref_rnn_node = std::make_shared<ngraph::op::Rnn>(rnn_src_layer,
                                                          rnn_src_iter,
#if MKLDNN_VERSION_MAJOR >= 1
                                                          rnn_src_iter_c,
#endif
                                                          rnn_weights_layer,
                                                          rnn_weights_iter,
                                                          rnn_bias,
                                                          ref_number_of_timesteps,
                                                          ref_number_of_gates_per_cell,
                                                          ref_src_seq_length,
                                                          ref_num_rnn_cell_states,
                                                          ref_rnn_direction,
                                                          ref_num_of_rnn_fused_layer,
                                                          ref_rnn_type);

    auto rnn_goe0 = std::make_shared<ngraph::op::GetOutputElement>(ref_rnn_node, 0);

    auto rnn_goe0_label =
        std::make_shared<pattern::op::Label>(rnn_goe0, nullptr, NodeVector{rnn_goe0});

    auto callback = [rnn_src_layer,
                     rnn_src_iter,
#if MKLDNN_VERSION_MAJOR >= 1
                     rnn_src_iter_c,
#endif
                     rnn_weights_layer,
                     rnn_weights_iter,
                     rnn_bias,
                     rnn_goe0_label](pattern::RecurrentMatcher& m) {
        auto number_of_rnn_cell_matched = m.get_number_of_recurrent_matches();
        NGRAPH_DEBUG << " In Recurrent multi layer RNN fusion callback ";
        NGRAPH_DEBUG << " Number of RNN's Matched: " << number_of_rnn_cell_matched;
        NGRAPH_DEBUG << " matched_root: " << m.get_match_root()->get_name();

        if (number_of_rnn_cell_matched < 2)
        {
            return false;
        }

        auto rnn_goe0_bounded_nodes = m.get_bound_nodes_for_pattern(rnn_goe0_label);

        std::vector<std::shared_ptr<ngraph::op::Rnn>> rnn_nodes;
        for (auto rnn_goe : m.get_bound_nodes_for_pattern(rnn_goe0_label))
        {
            if (auto rnn_op =
                    std::dynamic_pointer_cast<ngraph::op::Rnn>(rnn_goe->get_arguments()[0]))
            {
                rnn_nodes.push_back(rnn_op);
            }
            else
            {
                NGRAPH_DEBUG << "PM error, input to GOE is not RNN";
                return false;
            }
        }

        size_t num_timesteps = rnn_nodes[0]->get_num_timesteps();
        size_t lstm_n_gates = rnn_nodes[0]->get_gates_per_cell();
        size_t batch_size = rnn_nodes[0]->get_batch_size();
        size_t sequence_len = rnn_nodes[0]->get_src_sequence_length();
        size_t src_iter_feature_size = rnn_nodes[0]->get_src_iter_feature_size();
        size_t num_rnn_cell_states = rnn_nodes[0]->get_num_cell_states();
        size_t rnn_direction = rnn_nodes[0]->get_direction();
        size_t num_fused_rnn_layers = rnn_nodes.size();
        ngraph::runtime::cpu::rnn_utils::rnntype rnn_type = rnn_nodes[0]->get_rnn_type();

        for (auto rnn_node : rnn_nodes)
        {
            if ((rnn_node->get_num_timesteps() != num_timesteps) ||
                (rnn_node->get_gates_per_cell() != lstm_n_gates) ||
                (rnn_node->get_batch_size() != batch_size) ||
                (rnn_node->get_src_sequence_length() != sequence_len) ||
                (rnn_node->get_src_iter_feature_size() != src_iter_feature_size) ||
                (rnn_node->get_num_cell_states() != num_rnn_cell_states) ||
                (rnn_node->get_direction() != rnn_direction))
            {
                NGRAPH_DEBUG << "RNN attributes dont match";
                return false;
            }
            if (rnn_node->get_dst_layer_feature_size() != rnn_node->get_src_layer_feature_size())
            {
                // we will look at the matched RNN cells and only fuse the RNN if we have
                // SLC == DLC
                NGRAPH_DEBUG << "RNN SRC and DST feature sizes differ";
                return false;
            }
        }

        // the last matched rnn cell with slc=dlc will be in the input to the new fused
        // node, PM captures the RNN cell in the reverse order.
        // {RNN7, RNN6, RNN5.....RNN0}
        auto mrnn_src_layer =
            m.get_bound_nodes_for_pattern(rnn_src_layer)[number_of_rnn_cell_matched - 1];
        auto mrnn_src_iter = stack_rnn_inputs(m.get_bound_nodes_for_pattern(rnn_src_iter));
#if MKLDNN_VERSION_MAJOR >= 1
        auto mrnn_src_iter_c = stack_rnn_inputs(m.get_bound_nodes_for_pattern(rnn_src_iter_c));
#endif
        auto mrnn_weights_layer =
            stack_rnn_inputs(m.get_bound_nodes_for_pattern(rnn_weights_layer));
        auto mrnn_weights_iter = stack_rnn_inputs(m.get_bound_nodes_for_pattern(rnn_weights_iter));
        auto mrnn_bias = stack_rnn_inputs(m.get_bound_nodes_for_pattern(rnn_bias));

        NGRAPH_DEBUG << "src_layer: " << join(mrnn_src_layer->get_shape());
        NGRAPH_DEBUG << "src_iter: " << join(mrnn_src_iter->get_shape());
#if MKLDNN_VERSION_MAJOR >= 1
        NGRAPH_DEBUG << "src_iter_c: " << join(mrnn_src_iter_c->get_shape());
#endif
        NGRAPH_DEBUG << "weights_layer: " << join(mrnn_weights_layer->get_shape());
        NGRAPH_DEBUG << "weights_iter: " << join(mrnn_weights_iter->get_shape());
        NGRAPH_DEBUG << "bias: " << join(mrnn_bias->get_shape());
        NGRAPH_DEBUG << "src_seq_len: " << sequence_len;
        NGRAPH_DEBUG << "batch_size: " << batch_size;
        NGRAPH_DEBUG << "src iter feature_size: " << src_iter_feature_size;

        if ((mrnn_src_layer->get_shape()[0] / batch_size) != rnn_nodes[0]->get_num_timesteps())
        {
            throw ngraph_error(
                " input symbols for the layer fused RNN op, should be captured only for the first "
                "layer");
        }

        auto rnn = std::make_shared<ngraph::op::Rnn>(mrnn_src_layer,
                                                     mrnn_src_iter,
#if MKLDNN_VERSION_MAJOR >= 1
                                                     mrnn_src_iter_c,
#endif
                                                     mrnn_weights_layer,
                                                     mrnn_weights_iter,
                                                     mrnn_bias,
                                                     num_timesteps,
                                                     lstm_n_gates,
                                                     sequence_len,
                                                     num_rnn_cell_states,
                                                     rnn_direction,
                                                     num_fused_rnn_layers,
                                                     rnn_type);

        auto mrnn_ht = std::make_shared<ngraph::op::GetOutputElement>(rnn, 0);
#if MKLDNN_VERSION_MAJOR < 1
        auto mrnn_ht_ct = std::make_shared<ngraph::op::GetOutputElement>(rnn, 1);

        // Replace all the users of RNN cell state {ct} across different user.
        auto replace_rnn_output_cellstate = [&](std::shared_ptr<Node> rnn_ct_goe1, size_t layer) {

            // multi layerd fused rnn second output {GOE1} holds the recurrent output state tensors
            // for the last cell of all the layers, {{ht_1 | ct_1} || {ht2 |ct2} || ....{htn | ctn}}
            // we will slice the cell state output tensor {ct_*} from the fused RNN kerenel output
            // and feeds {ct_*} consumer if any
            auto ct_slice = std::make_shared<ngraph::op::Slice>(
                mrnn_ht_ct,
                Coordinate{((layer - 1) * batch_size * num_rnn_cell_states) + batch_size, 0},
                Coordinate{layer * batch_size * num_rnn_cell_states, src_iter_feature_size});

            replace_collapse_node_user(rnn_ct_goe1, ct_slice->output(0));
        };

        // we will replace cell_state {ct} of all the matched RNN cell with the new {ct} of the
        // fused RNN cell Note: RNN cells are captured in the reverse order i.e {RNN7, RNN6,
        // RNN5.... RNN0}
        for (size_t index = 0; index < rnn_nodes.size(); index++)
        {
            auto goe_nodes = ngraph::op::get_output_elements(rnn_nodes[index]);
            // if there is no GOE followed by the Lstm, their might be pattern match error we will
            // return safely
            if (goe_nodes.size() != 2)
            {
                throw ngraph_error("Expecting two outputs for each RNN node");
            }

            // dst_layer of the RNN cell
            auto goe_0 = goe_nodes[0];
            // dst_iter of the RNN cell
            auto goe_1 = goe_nodes[1];

            if (goe_1)
            {
                int layer_index = num_fused_rnn_layers - index;
                replace_rnn_output_cellstate(goe_1, layer_index);
            }

            // dst_layer of layer fused rnn holds the intermediate results of all the lstm cells
            // belonging to the last layer we will replace the GOE, since RNN_n->GOE0 and
            // MutliLayerRnn->GOE0 holds the same output
            if ((index == 0) && goe_0)
            {
                replace_collapse_node_user(goe_0, mrnn_ht->output(0));
            }
        }
#else
        auto mrnn_ct = std::make_shared<ngraph::op::GetOutputElement>(rnn, 2);

        // Replace all the users of RNN cell state {ct} across different user.
        auto replace_rnn_output_cellstate = [&](std::shared_ptr<Node> rnn_ct_goe2, size_t layer) {

            // multi layerd fused rnn second output {GOE2} holds the recurrent output state tensors
            // for the last cell
            // of all the layers, { ct_1 || ct2 || ....|| ctn}
            auto ct_slice = std::make_shared<ngraph::op::Slice>(
                mrnn_ct,
                Coordinate{((layer - 1) * batch_size) + batch_size, 0},
                Coordinate{layer * batch_size, src_iter_feature_size});

            replace_collapse_node_user(rnn_ct_goe2, ct_slice->output(0));
        };

        // we will replace cell_state {ct} of all the matched RNN cell
        // with the new {ct} of the fused RNN cell
        // Note: RNN cells are captured in the reverse order
        // i.e {RNN7, RNN6, RNN5.... RNN0}
        for (size_t index = 0; index < rnn_nodes.size(); index++)
        {
            auto goe_nodes = ngraph::op::get_output_elements(rnn_nodes[index]);
            // if there is no GOE followed by the Lstm, their might be pattern match error
            // we will return safely
            if (goe_nodes.size() != 3)
            {
                throw ngraph_error("Expecting three outputs for each RNN node");
            }

            // dst_layer of the RNN cell
            auto goe_0 = goe_nodes[0];
            // dst_iter of the RNN cell
            auto goe_1 = goe_nodes[1];
            // dst_iter_c of the RNN cell
            auto goe_2 = goe_nodes[2];

            if (goe_2)
            {
                int layer_index = num_fused_rnn_layers - index;
                replace_rnn_output_cellstate(goe_2, layer_index);
            }

            // dst_layer of layer fused rnn holds the intermediate results of all the lstm cells
            // belonging to the last layer we will replace the GOE, since RNN_n->GOE0 and
            // MutliLayerRnn->GOE0
            // holds the same output
            if ((index == 0) && goe_0)
            {
                replace_collapse_node_user(goe_0, mrnn_ht->output(0));
            }
        }
#endif
        return true;
    };

    std::set<std::shared_ptr<pattern::op::Label>> empty_correlated_matches;
    auto m = std::make_shared<pattern::RecurrentMatcher>(
        rnn_goe0_label, rnn_src_layer, empty_correlated_matches);
    this->add_matcher(m, callback);
}

void ngraph::runtime::cpu::pass::BiDirectionalRnn::construct_bidirectional_rnn()
{
    auto rnn_left_to_right = std::make_shared<pattern::op::Label>(
        element::f32, Shape{1, 256}, pattern::has_class<ngraph::op::Rnn>());
    auto rnn_right_to_left = std::make_shared<pattern::op::Label>(
        element::f32, Shape{1, 256}, pattern::has_class<ngraph::op::Rnn>());

    auto reshape_pred = [](std::shared_ptr<Node> n) {
        return (std::dynamic_pointer_cast<ngraph::op::Reshape>(n) != nullptr);
    };
    auto rnn_left_to_right_goe0 =
        std::make_shared<ngraph::op::GetOutputElement>(rnn_left_to_right, 0);
    auto rnn_right_to_left_goe0 =
        std::make_shared<ngraph::op::GetOutputElement>(rnn_right_to_left, 0);

    auto rnn_rtol_goe0_reshape_ntc =
        std::make_shared<pattern::op::Skip>(rnn_right_to_left_goe0, reshape_pred);
    auto rnn_rtol_goe0_reshape_tnc =
        std::make_shared<pattern::op::Skip>(rnn_rtol_goe0_reshape_ntc, reshape_pred);
    auto rnn_ltor_goe0_reshape_ntc =
        std::make_shared<pattern::op::Skip>(rnn_left_to_right_goe0, reshape_pred);
    auto rnn_ltor_goe0_reshape_tnc =
        std::make_shared<pattern::op::Skip>(rnn_ltor_goe0_reshape_ntc, reshape_pred);

    auto reverse_seq_predicate = [](std::shared_ptr<Node> node) {
        return pattern::has_class<ngraph::op::ReverseSequence>()(node) ||
               pattern::has_class<ngraph::op::Reverse>()(node);
    };
    auto skip_reverse_seq =
        std::make_shared<pattern::op::Skip>(rnn_rtol_goe0_reshape_tnc, reverse_seq_predicate);
    auto concat = std::make_shared<ngraph::op::Concat>(
        NodeVector{rnn_ltor_goe0_reshape_tnc, skip_reverse_seq}, 0);

    // Define a call back that needs to called once the DFG matches the pattern
    auto callback = [rnn_left_to_right, rnn_right_to_left](pattern::Matcher& m) {

        auto pattern_map = m.get_pattern_map();
        auto rnn_ltor_node =
            std::static_pointer_cast<ngraph::op::Rnn>(pattern_map[rnn_left_to_right]);
        auto rnn_rtol_node =
            std::static_pointer_cast<ngraph::op::Rnn>(pattern_map[rnn_right_to_left]);

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

        if (rnn_ltor_node->get_src_sequence_length() != rnn_rtol_node->get_src_sequence_length())
        {
            NGRAPH_DEBUG << " Not fusing, sequence length  of rnn's in both direction should match";
            return false;
        }

        size_t num_time_steps = rnn_ltor_node->get_num_timesteps();
        size_t lstm_n_gates = rnn_ltor_node->get_gates_per_cell();
        size_t sequence_len = rnn_ltor_node->get_src_sequence_length();
        size_t num_rnn_cell_states = rnn_ltor_node->get_num_cell_states();
        size_t rnn_direction = 2;
        size_t num_fused_rnn_layers = 1;
        ngraph::runtime::cpu::rnn_utils::rnntype rnn_type =
            ngraph::runtime::cpu::rnn_utils::rnntype::vanilla_lstm;

        auto construct_birnn_inputs = [&](int index) {

            auto nodes =
                NodeVector{rnn_ltor_node->get_argument(index), rnn_rtol_node->get_argument(index)};
            return std::make_shared<ngraph::op::Concat>(nodes, 0);
        };

        auto src_layer = rnn_ltor_node->get_arguments()[0];
        auto src_iter = construct_birnn_inputs(1);
#if MKLDNN_VERSION_MAJOR < 1
        auto weights_layer = construct_birnn_inputs(2);
        auto weights_iter = construct_birnn_inputs(3);
        auto bias = construct_birnn_inputs(4);

        auto rnn = std::make_shared<ngraph::op::Rnn>(src_layer,
                                                     src_iter,
                                                     weights_layer,
                                                     weights_iter,
                                                     bias,
                                                     num_time_steps,
                                                     lstm_n_gates,
                                                     sequence_len,
                                                     num_rnn_cell_states,
                                                     rnn_direction,
                                                     num_fused_rnn_layers,
                                                     rnn_type);
#else
        auto src_iter_c = construct_birnn_inputs(2);
        auto weights_layer = construct_birnn_inputs(3);
        auto weights_iter = construct_birnn_inputs(4);
        auto bias = construct_birnn_inputs(5);

        auto rnn = std::make_shared<ngraph::op::Rnn>(src_layer,
                                                     src_iter,
                                                     src_iter_c,
                                                     weights_layer,
                                                     weights_iter,
                                                     bias,
                                                     num_time_steps,
                                                     lstm_n_gates,
                                                     sequence_len,
                                                     num_rnn_cell_states,
                                                     rnn_direction,
                                                     num_fused_rnn_layers,
                                                     rnn_type);
#endif

        auto layer_rnn_ht = std::make_shared<ngraph::op::GetOutputElement>(rnn, 0);
        size_t batch_size = layer_rnn_ht->get_shape()[0] / num_time_steps;
        size_t feature_size = layer_rnn_ht->get_shape()[1];

        // if the shape doesnt match, we will logically reshape it to expaned_dims{tnc} from
        // squeezed_dims{t*n, c}
        std::shared_ptr<Node> layer_rnn_ht_reshape = layer_rnn_ht;
        if (m.get_match_root()->get_shape() != layer_rnn_ht->get_shape())
        {
            layer_rnn_ht_reshape = std::make_shared<ngraph::op::Reshape>(
                layer_rnn_ht, AxisVector{0, 1}, Shape{num_time_steps, batch_size, feature_size});
        }

        // we will check if the node being replaced is in Shape{n, t, c}, if so we will transpose
        if (m.get_match_root()->get_shape() == Shape{batch_size, num_time_steps, feature_size})
        {
            layer_rnn_ht_reshape = std::make_shared<ngraph::op::Reshape>(
                layer_rnn_ht_reshape,
                AxisVector{1, 0, 2},
                Shape{batch_size, num_time_steps, feature_size});
        }

        ngraph::replace_node(m.get_match_root(), layer_rnn_ht_reshape);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(concat, "BiDirectionalRnn");
    this->add_matcher(m, callback);
}
