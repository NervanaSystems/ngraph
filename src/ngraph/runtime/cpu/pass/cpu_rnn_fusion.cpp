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
#include "ngraph/op/lstm_cell.hpp"
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
#include "ngraph/pattern/op/or.hpp"
#include "ngraph/pattern/op/skip.hpp"
#include "ngraph/runtime/cpu/dnnl_utils.hpp"
#include "ngraph/runtime/cpu/op/lstm.hpp"
#include "ngraph/runtime/cpu/op/rnn.hpp"
#include "ngraph/runtime/cpu/op/rnn_utils.hpp"

#define STR(X) #X
#define CHECK_RANK(X, RANK)                                                                        \
    if (X.get_shape().size() != RANK)                                                              \
    {                                                                                              \
        NGRAPH_DEBUG << STR(X) << " does not have rank " << RANK;                                  \
        return false;                                                                              \
    }

#define CHECK_VALUE_RANK(X, RANK)                                                                  \
    if (X.get_shape().size() != RANK)                                                              \
    {                                                                                              \
        NGRAPH_DEBUG << STR(X) << " does not have rank " << RANK;                                  \
        return false;                                                                              \
    }

using namespace ngraph;

void ngraph::runtime::cpu::pass::VanillaRNNFusion::construct_vanilla_rnn()
{
    // pattern to capture the vanilla RNN
    // at = W*h{t, l-1} + U *h{t-1, l} + B
    // ht = activation(at)

    auto src_layer_label = std::make_shared<pattern::op::Label>(element::f32, Shape{32, 34});
    auto src_iter_label = std::make_shared<pattern::op::Label>(element::f32, Shape{32, 34});
    auto concat =
        std::make_shared<ngraph::op::v0::Concat>(OutputVector{src_layer_label, src_iter_label}, 0);
    auto weights = std::make_shared<pattern::op::Label>(element::f32, Shape{34, 2});
    auto bias_label = std::make_shared<pattern::op::Label>(element::f32, Shape{64, 2});
    auto broadcast_pred = [](Output<Node> n) {
        return ((is_type<ngraph::op::v0::Broadcast>(n.get_node())) ||
                (is_type<ngraph::op::v0::Reshape>(n.get_node())));
    };
    auto dot = std::make_shared<ngraph::op::v0::Dot>(concat, weights);
    auto add = std::make_shared<ngraph::op::v1::Add>(
        dot, std::make_shared<pattern::op::Skip>(bias_label, broadcast_pred));

    auto activation = std::make_shared<ngraph::op::v0::Tanh>(add);

    auto callback = [src_layer_label, src_iter_label, weights, bias_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In construct_vanilla_rnn callback against " << *m.get_match_root();
        auto pattern_map = m.get_pattern_map();
        ngraph::runtime::cpu::rnn_utils::rnntype rnn_type =
            ngraph::runtime::cpu::rnn_utils::rnntype::vanilla_rnn;

        auto fused_weights = pattern_map[weights];
        auto bias = pattern_map[bias_label];
        auto src_layer = pattern_map[src_layer_label];
        auto src_iter = pattern_map[src_iter_label];

        size_t slc = src_layer->get_output_shape(0)[1];
        size_t sic = src_iter->get_output_shape(0)[1];
        size_t dlc = fused_weights->get_output_shape(0)[1];
        size_t n_gates = 1;
        size_t direction = 1;
        size_t n_layers = 1;
        size_t n_state = 1;
        size_t time_steps = 1;
        size_t seq_length = 1;

        // split the fused weights for RNN kernel
        auto wei_layer = std::make_shared<ngraph::op::v0::Slice>(
            fused_weights, Coordinate{0, 0}, Coordinate{slc, dlc});
        auto wei_iter = std::make_shared<ngraph::op::v0::Slice>(
            fused_weights, Coordinate{slc, 0}, Coordinate{slc + sic, dlc});

        auto rnn_node = std::make_shared<ngraph::op::Rnn>(src_layer,
                                                          src_iter,
                                                          wei_layer,
                                                          wei_iter,
                                                          bias,
                                                          time_steps,
                                                          n_gates,
                                                          seq_length,
                                                          n_state,
                                                          direction,
                                                          n_layers,
                                                          rnn_type);

        m.get_match_value().replace(rnn_node->output(0));
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(activation, "VanillaRNNFusion.vanilla_rnn");
    this->add_matcher(m, callback);
}

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
    auto B = std::make_shared<pattern::op::Label>(element::f32,
                                                  Shape{ref_gates_count * ref_hidden_size});
    auto peep_hole = std::make_shared<pattern::op::Label>(element::f32, Shape{3 * ref_hidden_size});
    auto H_t =
        std::make_shared<pattern::op::Label>(element::f32, Shape{ref_batch_size, ref_hidden_size});
    auto C_t =
        std::make_shared<pattern::op::Label>(element::f32, Shape{ref_batch_size, ref_hidden_size});

    auto ref_lstm_cell =
        std::make_shared<op::v0::LSTMCell>(X,
                                           H_t,
                                           C_t,
                                           W,
                                           R,
                                           B,
                                           peep_hole,
                                           ref_hidden_size,
                                           op::LSTMWeightsFormat::IOFC,
                                           std::vector<std::string>{"sigmoid", "tanh", "tanh"},
                                           std::vector<float>{},
                                           std::vector<float>{},
                                           0.f,
                                           false);

    auto callback = [X, W, R, H_t, C_t](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In construct_onnx_lstmcell_fprop callback against " << *m.get_match_root();
        auto pattern_map = m.get_pattern_map();
        ngraph::runtime::cpu::rnn_utils::rnntype rnn_type =
            ngraph::runtime::cpu::rnn_utils::rnntype::vanilla_lstm;

        auto lstmcell_op = m.get_match_root_as<op::v0::LSTMCell>();
        NGRAPH_CHECK(lstmcell_op,
                     "match root node ",
                     *m.get_match_root(),
                     " not of type `op::v0::LSTMCell`");
        auto src_iter = std::make_shared<ngraph::op::v0::Concat>(
            OutputVector{pattern_map[H_t], pattern_map[C_t]}, 0);

        auto W_ifco = lstmcell_op->get_argument(3);
        auto R_ifco = lstmcell_op->get_argument(4);
        auto bias_ifco = lstmcell_op->get_argument(5);

        // We need to reorder W, R and bias to IFCO gate order.
        // Note: ie.: ONNX runtime provides W, R and bias in the gate order [IOFC] but
        // DNNL computes LSTM kernel in the [IFCO] order.
        if (lstmcell_op->get_weights_format() != op::LSTMWeightsFormat::IFCO)
        {
            W_ifco = lstmcell_op->convert_node_format(W_ifco);
            R_ifco = lstmcell_op->convert_node_format(R_ifco);
            bias_ifco = lstmcell_op->convert_node_format(bias_ifco);
        }

        auto W_reshape = std::make_shared<op::v0::Reshape>(
            W_ifco,
            AxisVector{1, 0},
            Shape{W_ifco->get_output_shape(0)[1], W_ifco->get_output_shape(0)[0]});
        auto R_reshape = std::make_shared<op::v0::Reshape>(
            R_ifco,
            AxisVector{1, 0},
            Shape{R_ifco->get_output_shape(0)[1], R_ifco->get_output_shape(0)[0]});

        auto lstm_node = std::make_shared<ngraph::op::Lstm>(pattern_map[X],
                                                            pattern_map[H_t],
                                                            pattern_map[C_t],
                                                            W_reshape,
                                                            R_reshape,
                                                            bias_ifco,
                                                            rnn_type);
        if (lstm_node->get_output_size() != 3)
        {
            throw ngraph_error("Lstm node doesnt have three outputs");
        }

        auto dst_layer = m.get_match_value().get_node()->output(0);
        auto dst_iter = m.get_match_value().get_node()->output(1);
        // dst_iter of lstm dnnl output holds the results of both recurrent state
        // tensor outputs. we need to slice the ct.
        // find the user's for {ht} and replace them with lstm output 2
        dst_iter.replace(lstm_node->output(2));
        // find the user's for {ht} and replace them with lstm output 1
        dst_layer.replace(lstm_node->output(1));
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(ref_lstm_cell, "LSTMFusion.onnx_lstm_cell");
    this->add_matcher(m, callback);
}

void ngraph::runtime::cpu::pass::LSTMFusion::construct_sigmoid()
{
    // construct variance
    auto input = std::make_shared<pattern::op::Label>(element::f32, Shape{3, 4});
    auto neg_input = std::make_shared<ngraph::op::v0::Negative>(input);
    auto exp_neg_input = std::make_shared<ngraph::op::v0::Exp>(neg_input);

    // broadcast input
    auto constant = std::make_shared<pattern::op::Label>(element::f32, Shape{});
    auto broadcast_constant =
        std::make_shared<ngraph::op::v0::Broadcast>(constant, Shape{3, 4}, AxisSet{0, 1});

    auto add_exp = std::make_shared<ngraph::op::v1::Add>(exp_neg_input, broadcast_constant);
    auto divide_1_over_exp = std::make_shared<ngraph::op::v1::Divide>(broadcast_constant, add_exp);

    // Define a call back that needs to called once the DFG matches the pattern
    auto callback = [input](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In construct_sigmoid pattern callback against " << *m.get_match_root();

        auto pattern_map = m.get_pattern_map();

        if (m.get_match_value().get_element_type() != element::f32)
        {
            NGRAPH_DEBUG << "mpattern = " << m.get_match_root()->get_name()
                         << " type is not float!";
            return false;
        }

        if (m.get_match_root()->get_output_size() != pattern_map[input]->get_output_size())
        {
            NGRAPH_DEBUG << "mpattern = " << m.get_match_root()->get_name()
                         << "input= " << pattern_map[input]->get_name() << "size dont match!";
            return false;
        }

        auto sigmoid_node = std::make_shared<ngraph::op::v0::Sigmoid>(pattern_map[input]);
        m.get_match_value().replace(sigmoid_node->output(0));
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(divide_1_over_exp, "LSTMFusion.Sigmoid");
    this->add_matcher(m, callback);
}

static void replace_collapse_node_user(const Output<Node>& collapsed_node,
                                       const Output<Node>& new_output)
{
    for (Input<Node> input : collapsed_node.get_target_inputs())
    {
        input.replace_source_output(new_output);
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

    auto broadcast_pred = [](Output<Node> n) {
        return ((is_type<ngraph::op::v0::Broadcast>(n.get_node())) ||
                (is_type<ngraph::op::v0::Reshape>(n.get_node())));
    };

    // Fused MatMuls
    // (W_{ii} | (W_{if} | W_{ig} | W_{io}) * x_t + (b_{ii} | b_{if} |  b_{ig} | b_{io})
    auto dot1 = std::make_shared<ngraph::op::v0::Dot>(xt, w_i2h);
    auto add1 = std::make_shared<ngraph::op::v1::Add>(
        dot1, std::make_shared<pattern::op::Skip>(bias_i2h, broadcast_pred));
    // (W_{hi} | (W_{hf} | W_{hg} | W_{ho}) * h_{(t-1)} + (b_{hi} | b_{hf} |  b_{hg} | b_{ho})
    auto dot2 = std::make_shared<ngraph::op::v0::Dot>(ht_1, w_h2h);
    auto add2 = std::make_shared<ngraph::op::v1::Add>(
        dot2, std::make_shared<pattern::op::Skip>(bias_h2h, broadcast_pred));

    auto X = std::make_shared<ngraph::op::v1::Add>(add2, add1);

    // construct gates
    auto it = std::make_shared<ngraph::op::v0::Sigmoid>(
        std::make_shared<ngraph::op::v0::Slice>(X, Coordinate{0, 0}, Coordinate{10, 100}));
    auto ft = std::make_shared<ngraph::op::v0::Sigmoid>(
        std::make_shared<ngraph::op::v0::Slice>(X, Coordinate{0, 100}, Coordinate{10, 200}));
    auto gt = std::make_shared<ngraph::op::v0::Tanh>(
        std::make_shared<ngraph::op::v0::Slice>(X, Coordinate{0, 200}, Coordinate{10, 300}));
    auto ot = std::make_shared<ngraph::op::v0::Sigmoid>(
        std::make_shared<ngraph::op::v0::Slice>(X, Coordinate{0, 300}, Coordinate{10, 400}));

    // construct (c_t) cell state
    auto ct =
        std::make_shared<ngraph::op::v1::Add>(std::make_shared<ngraph::op::v1::Multiply>(ft, ct_1),
                                              std::make_shared<ngraph::op::v1::Multiply>(it, gt));
    auto ct_label = std::make_shared<pattern::op::Label>(ct, nullptr, OutputVector{ct});

    // construct (h_t)
    auto ht = std::make_shared<ngraph::op::v1::Multiply>(
        ot, std::make_shared<ngraph::op::v0::Tanh>(ct_label));

    // Define a call back that needs to called once the DFG matches the pattern
    auto callback = [this, ct_label, w_i2h, bias_i2h, w_h2h, bias_h2h, xt, ht_1, ct_1](
                        pattern::Matcher& m) {
        NGRAPH_DEBUG << "In construct_lstm_fprop callback against " << *m.get_match_root();

        auto pattern_map = m.get_pattern_value_map();

        if (m.get_match_value().get_element_type() != element::f32)
        {
            NGRAPH_DEBUG << "mpattern type is not float!";
            return false;
        }

        CHECK_VALUE_RANK(pattern_map[xt], 2)
        CHECK_VALUE_RANK(pattern_map[ht_1], 2)
        CHECK_VALUE_RANK(pattern_map[w_i2h], 2)
        CHECK_VALUE_RANK(pattern_map[w_h2h], 2)
        CHECK_VALUE_RANK(pattern_map[bias_i2h], 1)
        CHECK_VALUE_RANK(pattern_map[bias_h2h], 1)

        auto weights_layer = pattern_map[w_i2h];
        auto weights_iter = pattern_map[w_h2h];
        auto src_layer = pattern_map[xt];
        auto hidden_state = pattern_map[ht_1];
        auto cell_state = pattern_map[ct_1];

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
        if (is_type<ngraph::op::v0::Broadcast>(src_layer.get_node_shared_ptr()) &&
            is_type<ngraph::op::v0::Constant>(
                src_layer.get_node_shared_ptr()->input_value(0).get_node_shared_ptr()))
        {
            // First timestep of an RNN layer
            swap_lstm_inputs();
        }
        else if (hidden_state.get_shape() != cell_state.get_shape())
        {
            swap_lstm_inputs();
        }
        else if (hidden_state.get_node() != cell_state.get_node())
        {
            swap_lstm_inputs();
        }

        if (hidden_state.get_shape() != cell_state.get_shape())
        {
            NGRAPH_DEBUG << "Lstm DNNL kernel requires recurrent output hidden states to match ";
            return false;
        }

        // set LSTM cell attributes
        size_t lstm_n_gates = 4;
        size_t direction = 1;
        size_t layers = 1;
        auto dlc = weights_layer.get_shape()[1] / (lstm_n_gates * direction * layers);
        auto slc = weights_layer.get_shape()[0];
        auto dic = weights_iter.get_shape()[1] / (lstm_n_gates * direction * layers);
        auto sic = weights_iter.get_shape()[0];
        ngraph::runtime::cpu::rnn_utils::rnntype rnn_type =
            ngraph::runtime::cpu::rnn_utils::rnntype::vanilla_lstm;

        if (dlc != dic)
        {
            NGRAPH_DEBUG << "Not fusing, since Lstm kernel requires dst_layer feature size "
                         << "equals to dts_iter feature size";
            return false;
        }

        auto bias =
            std::make_shared<ngraph::op::v1::Add>(pattern_map[bias_i2h], pattern_map[bias_h2h]);

        if (src_layer.get_shape()[1] != slc || hidden_state.get_shape()[1] != sic ||
            cell_state.get_shape()[1] != sic)
        {
            NGRAPH_DEBUG << "Feature size mismatch between weights and input tensors";
            return false;
        }
        auto lstm_node = std::make_shared<ngraph::op::Lstm>(
            src_layer, hidden_state, cell_state, weights_layer, weights_iter, bias, rnn_type);

        auto lstm_ht_output = lstm_node->output(1);
        auto lstm_ct_output = lstm_node->output(2);

        // Now identify the nodes which consumes the output of LSTM nodes
        // and replace them accordingly
        // find the user's for {ct} and replace them with lstm_goe_2
        if (ngraph::is_used(pattern_map[ct_label].get_node()))
        {
            replace_collapse_node_user(pattern_map[ct_label], lstm_ct_output);
        }
        // find the user's for {ht} and replace them with lstm_goe_1
        m.get_match_value().replace(lstm_ht_output);
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

    auto lstm_weights_layer_shared = std::make_shared<pattern::op::Label>(
        element::f32, Shape{400, 100}, pattern::has_class<ngraph::op::v0::Parameter>());
    auto lstm_weights_layer = std::make_shared<ngraph::op::v0::Reshape>(
        lstm_weights_layer_shared, AxisVector{1, 0}, Shape{100, 400});
    auto lstm_weights_layer_label = std::make_shared<pattern::op::Label>(
        lstm_weights_layer, nullptr, OutputVector{lstm_weights_layer});

    auto lstm_weights_iter_shared = std::make_shared<pattern::op::Label>(
        element::f32, Shape{400, 100}, pattern::has_class<ngraph::op::v0::Parameter>());
    auto lstm_weights_iter = std::make_shared<ngraph::op::v0::Reshape>(
        lstm_weights_iter_shared, AxisVector{1, 0}, Shape{100, 400});
    auto lstm_weights_iter_label = std::make_shared<pattern::op::Label>(
        lstm_weights_iter, nullptr, OutputVector{lstm_weights_iter});

    auto lstm_bias_layer_shared = std::make_shared<pattern::op::Label>(element::f32, Shape{400});
    auto lstm_bias_iter_shared = std::make_shared<pattern::op::Label>(element::f32, Shape{400});
    auto lstm_bias =
        std::make_shared<ngraph::op::v1::Add>(lstm_bias_layer_shared, lstm_bias_iter_shared);
    auto lstm_bias_label =
        std::make_shared<pattern::op::Label>(lstm_bias, nullptr, OutputVector{lstm_bias});
    ngraph::runtime::cpu::rnn_utils::rnntype ref_rnn_type =
        ngraph::runtime::cpu::rnn_utils::rnntype::vanilla_lstm;

    auto lstm = std::make_shared<ngraph::op::Lstm>(lstm_src_layer,
                                                   lstm_ht,
                                                   lstm_ct,
                                                   lstm_weights_layer_label,
                                                   lstm_weights_iter_label,
                                                   lstm_bias_label,
                                                   ref_rnn_type);

    auto m = std::make_shared<pattern::RecurrentMatcher>(
        lstm->output(1),
        lstm->output(2),
        lstm_ct,
        std::set<std::shared_ptr<pattern::op::Label>>{lstm_weights_layer_shared,
                                                      lstm_weights_iter_shared,
                                                      lstm_bias_layer_shared,
                                                      lstm_bias_iter_shared});

    auto callback = [this,
                     lstm_src_layer,
                     lstm_ht,
                     lstm_ct,
                     lstm_weights_layer_label,
                     lstm_weights_iter_label,
                     lstm_bias_label](pattern::RecurrentMatcher& m) {
        NGRAPH_DEBUG << "In construct_rnn_lstm_fprop callback against " << *m.get_match_root();

        auto concat_rnn_inputs_across_timestep =
            [&](std::shared_ptr<pattern::op::Label> input_label) -> std::shared_ptr<Node> {
            NodeVector concat_args;
            // src_layer -> concatenate input symbols from different LSTM cells belonging to same
            // RNN layer
            // in the order 0, 1, 2... t time slice
            {
                auto node_labels = m.get_bound_values_for_pattern(input_label);
                std::reverse(node_labels.begin(), node_labels.end());
                return std::make_shared<ngraph::op::v0::Concat>(node_labels, 0);
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
        auto rnn_src_iter = m.get_bound_values_for_pattern(lstm_ht)[sequence_len - 1];
        // pick src_iter_c from first lstm
        auto rnn_src_iter_c = m.get_bound_values_for_pattern(lstm_ct)[sequence_len - 1];
        // weights and bias are shared across lstms. so pick any
        auto rnn_weights_layer = m.get_bound_values_for_pattern(lstm_weights_layer_label)[0];
        auto rnn_weights_iter = m.get_bound_values_for_pattern(lstm_weights_iter_label)[0];
        auto rnn_bias = m.get_bound_values_for_pattern(lstm_bias_label)[0];

        const size_t lstm_n_gates = 4;
        const size_t batch_size = rnn_src_layer->get_output_shape(0)[0] / sequence_len;
        const size_t src_iter_feature_size = rnn_weights_iter.get_shape()[0];
        const size_t num_cell_states = 2;
        const size_t direction = 1;
        const size_t num_fused_rnn_layers = 1;
        ngraph::runtime::cpu::rnn_utils::rnntype rnn_type =
            ngraph::runtime::cpu::rnn_utils::rnntype::vanilla_lstm;

        NGRAPH_DEBUG << "src_layer shape: " << rnn_src_layer->get_output_shape(0);
        NGRAPH_DEBUG << "src_iter shape: " << rnn_src_iter.get_shape();
        NGRAPH_DEBUG << "src_iter_c shape: " << rnn_src_iter_c.get_shape();
        NGRAPH_DEBUG << "weights_layer shape: " << rnn_weights_layer.get_shape();
        NGRAPH_DEBUG << "weights_iter shape: " << rnn_weights_iter.get_shape();
        NGRAPH_DEBUG << "bias shape: " << rnn_bias.get_shape();
        NGRAPH_DEBUG << "src_seq_len: " << sequence_len;
        NGRAPH_DEBUG << "batch_size: " << batch_size;

        auto check_const_input = [&](std::shared_ptr<Node> n) {
            if (is_type<ngraph::op::v0::Constant>(n) ||
                (is_type<ngraph::op::v0::Broadcast>(n) &&
                 is_type<ngraph::op::v0::Constant>(n->get_argument(0))))
            {
                return true;
            }
            return false;
        };

        if (!check_const_input(rnn_src_iter.get_node()->get_argument(0)) ||
            !check_const_input(rnn_src_iter_c.get_node()->get_argument(0)))
        {
            NGRAPH_DEBUG << "Non const input for RNN state initializer";
            return false;
        }

        CHECK_RANK(rnn_src_layer->output(0), 2);
        CHECK_RANK(rnn_src_iter, 2);
        CHECK_RANK(rnn_src_iter_c, 2);
        CHECK_RANK(rnn_weights_layer, 2);
        CHECK_RANK(rnn_weights_iter, 2);
        CHECK_RANK(rnn_bias, 1);

        if (rnn_src_layer->get_output_element_type(0) != element::f32 ||
            rnn_src_iter.get_element_type() != element::f32 ||
            rnn_src_iter_c.get_element_type() != element::f32)
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

        std::vector<std::shared_ptr<ngraph::op::v0::Slice>> ht_slice_per_timestep(sequence_len,
                                                                                  nullptr);

        for (size_t i = 0, start_index = 0; i < sequence_len; i++, start_index += batch_size)
        {
            ht_slice_per_timestep[i] = (std::make_shared<ngraph::op::v0::Slice>(
                rnn->output(0),
                Coordinate{start_index, 0},
                Coordinate{start_index + batch_size, src_iter_feature_size}));
        }

        // find the lstm's nodes captured in PM
        auto lstm_cts = m.get_bound_values_for_pattern(lstm_ct);
        std::reverse(lstm_cts.begin(), lstm_cts.end());
        NodeVector lstm_nodes;

        // we need to collect LSTM from GOE's, in order to deterministically determine
        // the individual time slice output ht.
        for (size_t i = 0; i < sequence_len; i++)
        {
            // lstm's will be the user of lstm_ct
            for (auto user : lstm_cts[i].get_users())
            {
                if (is_type<ngraph::op::Lstm>(user))
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
            auto lstm = lstm_nodes[index];

            // if there is no GOE followed by the Lstm, their might be pattern match error
            // we will return safely
            if (!is_type<op::Lstm>(lstm))
            {
                return false;
            }

            // dst_iter of the lstm cell
            auto lstm1 = lstm->output(1);
            {
                for (std::shared_ptr<Node> lstm1_user : lstm1.get_users())
                {
                    // do not include LSTM in the same layer
                    if (std::find(lstm_nodes.begin(), lstm_nodes.end(), lstm1_user) ==
                        lstm_nodes.end())
                    {
                        for (size_t i = 0; i < lstm1_user->get_input_size(); i++)
                        {
                            if (lstm1_user->input_value(i) == lstm1)
                            {
                                lstm1_user->input(i).replace_source_output(
                                    ht_slice_per_timestep[index]->output(0));
                            }
                        }
                        NGRAPH_DEBUG << "ht_slice: " << ht_slice_per_timestep[index]->get_name()
                                     << " lstm1_user " << lstm1_user->get_name();
                    }
                }
            }
        }

        NGRAPH_DEBUG << "End of recurrent fusion call back "
                     << "matched_node: " << *m.get_match_root();
        return true;
    };

    this->add_matcher(m, callback);
}

static std::shared_ptr<Node> stack_rnn_inputs(OutputVector rnn_input_nodes)
{
    std::reverse(rnn_input_nodes.begin(), rnn_input_nodes.end());
    return std::make_shared<ngraph::op::v0::Concat>(rnn_input_nodes, 0);
}

void ngraph::runtime::cpu::pass::MultiLayerRNNFusion::construct_multi_layer_rnn_fusion_fprop()
{
    auto rnn_src_layer = std::make_shared<pattern::op::Label>(element::f32, Shape{30, 100});
    auto rnn_src_iter = std::make_shared<pattern::op::Label>(element::f32, Shape{10, 100});
    auto rnn_src_iter_c = std::make_shared<pattern::op::Label>(element::f32, Shape{10, 100});
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
                                                          rnn_src_iter_c,
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

    auto rnn_label = std::make_shared<pattern::op::Label>(
        ref_rnn_node->output(0), nullptr, ref_rnn_node->outputs());

    auto callback = [this,
                     rnn_src_layer,
                     rnn_src_iter,
                     rnn_src_iter_c,
                     rnn_weights_layer,
                     rnn_weights_iter,
                     rnn_bias,
                     rnn_label](pattern::RecurrentMatcher& m) {
        auto number_of_rnn_cell_matched = m.get_number_of_recurrent_matches();
        NGRAPH_DEBUG << "In construct_multi_layer_rnn_fusion_fprop callback against "
                     << *m.get_match_root();
        NGRAPH_DEBUG << "Number of RNN's Matched: " << number_of_rnn_cell_matched;

        if (number_of_rnn_cell_matched < 2)
        {
            return false;
        }

        auto rnn_bounded_nodes = m.get_bound_values_for_pattern(rnn_label);

        std::vector<std::shared_ptr<ngraph::op::Rnn>> rnn_nodes;
        for (auto rnn_goe : m.get_bound_values_for_pattern(rnn_label))
        {
            if (auto rnn_op = as_type_ptr<ngraph::op::Rnn>(rnn_goe.get_node_shared_ptr()))
            {
                rnn_nodes.push_back(rnn_op);
            }
            else if (auto rnn_op =
                         as_type_ptr<ngraph::op::Rnn>(rnn_goe.get_node()->get_arguments()[0]))
            {
                // This is a hack to support GOE on output of RNN
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
            m.get_bound_values_for_pattern(rnn_src_layer)[number_of_rnn_cell_matched - 1];
        auto mrnn_src_iter = stack_rnn_inputs(m.get_bound_values_for_pattern(rnn_src_iter));
        auto mrnn_src_iter_c = stack_rnn_inputs(m.get_bound_values_for_pattern(rnn_src_iter_c));
        auto mrnn_weights_layer =
            stack_rnn_inputs(m.get_bound_values_for_pattern(rnn_weights_layer));
        auto mrnn_weights_iter = stack_rnn_inputs(m.get_bound_values_for_pattern(rnn_weights_iter));
        auto mrnn_bias = stack_rnn_inputs(m.get_bound_values_for_pattern(rnn_bias));

        NGRAPH_DEBUG << "src_layer: " << join(mrnn_src_layer.get_shape());
        NGRAPH_DEBUG << "src_iter: " << join(mrnn_src_iter->get_output_shape(0));
        NGRAPH_DEBUG << "src_iter_c: " << join(mrnn_src_iter_c->get_output_shape(0));
        NGRAPH_DEBUG << "weights_layer: " << join(mrnn_weights_layer->get_output_shape(0));
        NGRAPH_DEBUG << "weights_iter: " << join(mrnn_weights_iter->get_output_shape(0));
        NGRAPH_DEBUG << "bias: " << join(mrnn_bias->get_output_shape(0));
        NGRAPH_DEBUG << "src_seq_len: " << sequence_len;
        NGRAPH_DEBUG << "batch_size: " << batch_size;
        NGRAPH_DEBUG << "src iter feature_size: " << src_iter_feature_size;

        if ((mrnn_src_layer.get_shape()[0] / batch_size) != rnn_nodes[0]->get_num_timesteps())
        {
            throw ngraph_error(
                " input symbols for the layer fused RNN op, should be captured only for the first "
                "layer");
        }

        auto rnn = std::make_shared<ngraph::op::Rnn>(mrnn_src_layer,
                                                     mrnn_src_iter,
                                                     mrnn_src_iter_c,
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

        // Replace all the users of RNN cell state {ct} across different user.
        auto replace_rnn_output_cellstate = [&](const Output<Node>& rnn_ct_goe2, size_t layer) {
            // multi layerd fused rnn second output {GOE2} holds the recurrent output state tensors
            // for the last cell
            // of all the layers, { ct_1 || ct2 || ....|| ctn}
            auto ct_slice = std::make_shared<ngraph::op::v0::Slice>(
                rnn->output(2),
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
            std::shared_ptr<Node> rnn_node = rnn_nodes[index];

            int layer_index = num_fused_rnn_layers - index;
            replace_rnn_output_cellstate(rnn_node->output(2), layer_index);

            // dst_layer of layer fused rnn holds the intermediate results of all the lstm cells
            // belonging to the last layer we will replace the GOE, since RNN_n->GOE0 and
            // MutliLayerRnn->GOE0
            // holds the same output
            if (index == 0)
            {
                replace_collapse_node_user(rnn_node->output(0), rnn->output(0));
            }
        }
        return true;
    };

    std::set<std::shared_ptr<pattern::op::Label>> empty_correlated_matches;
    auto m = std::make_shared<pattern::RecurrentMatcher>(
        rnn_label, rnn_src_layer, empty_correlated_matches);
    this->add_matcher(m, callback);
}

void ngraph::runtime::cpu::pass::BiDirectionalRnn::construct_bidirectional_rnn()
{
    auto rnn_left_to_right = std::make_shared<pattern::op::Label>(
        element::f32, Shape{1, 256}, pattern::has_class<ngraph::op::Rnn>());
    auto rnn_right_to_left = std::make_shared<pattern::op::Label>(
        element::f32, Shape{1, 256}, pattern::has_class<ngraph::op::Rnn>());

    auto reshape_pred = [](Output<Node> n) {
        return (is_type<ngraph::op::v0::Reshape>(n.get_node()));
    };

    auto rnn_rtol_reshape_ntc =
        std::make_shared<pattern::op::Skip>(rnn_right_to_left->output(0), reshape_pred);
    auto rnn_rtol_reshape_tnc =
        std::make_shared<pattern::op::Skip>(rnn_rtol_reshape_ntc, reshape_pred);
    auto rnn_ltor_reshape_ntc =
        std::make_shared<pattern::op::Skip>(rnn_left_to_right->output(0), reshape_pred);
    auto rnn_ltor_reshape_tnc =
        std::make_shared<pattern::op::Skip>(rnn_ltor_reshape_ntc, reshape_pred);

    auto reverse_seq_predicate = [](Output<Node> node) {
        return pattern::has_class<ngraph::op::v0::ReverseSequence>()(node) ||
               pattern::has_class<ngraph::op::v0::Reverse>()(node);
    };
    auto skip_reverse_seq =
        std::make_shared<pattern::op::Skip>(rnn_rtol_reshape_tnc, reverse_seq_predicate);
    auto concat = std::make_shared<ngraph::op::v0::Concat>(
        OutputVector{rnn_ltor_reshape_tnc, skip_reverse_seq}, 0);

    // Define a call back that needs to called once the DFG matches the pattern
    auto callback = [rnn_left_to_right, rnn_right_to_left](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In construct_bidirectional_rnn callback against " << *m.get_match_root();
        auto pattern_map = m.get_pattern_map();
        auto rnn_ltor_node = as_type_ptr<ngraph::op::Rnn>(pattern_map[rnn_left_to_right]);
        auto rnn_rtol_node = as_type_ptr<ngraph::op::Rnn>(pattern_map[rnn_right_to_left]);

        if (rnn_ltor_node->get_src_sequence_length() != rnn_rtol_node->get_src_sequence_length())
        {
            NGRAPH_DEBUG << "Not fusing, timestep of rnn's in both direction should match";
            return false;
        }

        if (rnn_ltor_node->get_src_layer_feature_size() !=
            rnn_rtol_node->get_src_layer_feature_size())
        {
            NGRAPH_DEBUG << "Not fusing, feature_size of rnn's in both direction should match";
            return false;
        }

        if (rnn_ltor_node->get_src_iter_feature_size() !=
            rnn_rtol_node->get_src_iter_feature_size())
        {
            NGRAPH_DEBUG << "Not fusing, feature_size of rnn's in both direction should match";
            return false;
        }

        if (rnn_ltor_node->get_batch_size() != rnn_rtol_node->get_batch_size())
        {
            NGRAPH_DEBUG << "Not fusing, feature_size of rnn's in both direction should match";
            return false;
        }

        if (rnn_ltor_node->get_src_sequence_length() != rnn_rtol_node->get_src_sequence_length())
        {
            NGRAPH_DEBUG << "Not fusing, sequence length  of rnn's in both direction should match";
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
            auto nodes = OutputVector{rnn_ltor_node->get_argument(index),
                                      rnn_rtol_node->get_argument(index)};
            return std::make_shared<ngraph::op::v0::Concat>(nodes, 0);
        };

        auto src_layer = rnn_ltor_node->get_arguments()[0];
        auto src_iter = construct_birnn_inputs(1);
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

        size_t batch_size = rnn->get_output_shape(0)[0] / num_time_steps;
        size_t feature_size = rnn->get_output_shape(0)[1];

        // if the shape doesnt match, we will logically reshape it to expaned_dims{tnc} from
        // squeezed_dims{t*n, c}
        Output<Node> layer_rnn_ht_reshape = rnn->output(0);
        if (m.get_match_value().get_shape() != rnn->get_output_shape(0))
        {
            layer_rnn_ht_reshape = std::make_shared<ngraph::op::v0::Reshape>(
                                       rnn->output(0),
                                       AxisVector{0, 1},
                                       Shape{num_time_steps, batch_size, feature_size})
                                       ->output(0);
        }

        // we will check if the node being replaced is in Shape{n, t, c}, if so we will transpose
        if (m.get_match_value().get_shape() == Shape{batch_size, num_time_steps, feature_size})
        {
            layer_rnn_ht_reshape = std::make_shared<ngraph::op::v0::Reshape>(
                                       layer_rnn_ht_reshape,
                                       AxisVector{1, 0, 2},
                                       Shape{batch_size, num_time_steps, feature_size})
                                       ->output(0);
        }

        m.get_match_value().replace(layer_rnn_ht_reshape);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(concat, "BiDirectionalRnn");
    this->add_matcher(m, callback);
}
