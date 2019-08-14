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

#pragma once

#include "ngraph/builder/split.hpp"
#include "ngraph/frontend/onnx_import/utils/reshape.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/fused/lstm_cell.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/reverse_sequence.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/util/broadcasting.hpp"

enum class LSTMDirection
{
    LSTM_DIRECTION_FORWARD,
    LSTM_DIRECTION_REVERSE,
    LSTM_DIRECTION_BIDIRECTIONAL,
    LSTM_DIRECTION_UNKNOWN,
};

namespace ngraph
{
    namespace op
    {
        ///
        /// \brief      Class for lstm sequence node.
        ///
        /// \note       It follows notation and equations defined as in ONNX standard:
        ///             https://github.com/onnx/onnx/blob/master/docs/Operators.md#LSTM
        ///
        class LSTMForward
        {
        public:
            explicit LSTMForward(const std::shared_ptr<Node>& X,
                                 const std::shared_ptr<Node>& W,
                                 const std::shared_ptr<Node>& R,
                                 const std::shared_ptr<Node>& B,
                                 const std::shared_ptr<Node>& P,
                                 const std::shared_ptr<Node>& initial_h,
                                 const std::shared_ptr<Node>& initial_c,
                                 const std::shared_ptr<Node>& seq_lengths,
                                 const std::vector<float> activations_alpha,
                                 const std::vector<float> activations_beta,
                                 const std::vector<std::string> activations,
                                 const float clip_threshold,
                                 const LSTMDirection direction,
                                 const std::int64_t hidden_size,
                                 const bool input_forget)
                : m_X{X}
                , m_W(W)
                , m_R(R)
                , m_B(B)
                , m_P(P)
                , m_initial_h(initial_h)
                , m_initial_c(initial_c)
                , m_seq_lengths(seq_lengths)
                , m_activations_alpha(activations_alpha)
                , m_activations_beta(activations_beta)
                , m_activations(activations)
                , m_clip_threshold(clip_threshold)
                , m_direction(direction)
                , m_hidden_size(hidden_size)
                , m_input_forget(input_forget)
            {
            }

            NodeVector run()
            {
                NodeVector results;

                if (m_direction == LSTMDirection::LSTM_DIRECTION_FORWARD ||
                    m_direction == LSTMDirection::LSTM_DIRECTION_REVERSE)
                {
                    results = lstm_pass(m_direction == LSTMDirection::LSTM_DIRECTION_REVERSE);
                }
                if (m_direction == LSTMDirection::LSTM_DIRECTION_BIDIRECTIONAL)
                {
                    NodeVector fwd_results{lstm_pass()};
                    NodeVector rev_results{lstm_pass(true)};

                    // Stack together respective outputs from both forward and reverse passess.
                    std::shared_ptr<Node> Y{std::make_shared<op::Concat>(
                        NodeVector{fwd_results.at(0), rev_results.at(0)}, 1)};
                    std::shared_ptr<Node> Y_h{std::make_shared<op::Concat>(
                        NodeVector{fwd_results.at(1), rev_results.at(1)}, 0)};
                    std::shared_ptr<Node> Y_c{std::make_shared<op::Concat>(
                        NodeVector{fwd_results.at(2), rev_results.at(2)}, 0)};
                    results = NodeVector{Y, Y_h, Y_c};
                }

                return results;
            }

        private:
            ///
            /// \brief      Gets the masked node according to sequence lenght in a batch.
            ///
            /// \note       Zeros out values or sets them to default value for inputs with
            ///             sequence lenght shorter than currently procssed time step.
            ///
            /// \param[in]  data           The input node.
            /// \param[in]  time_step      The current time step denoting sequence lenght.
            /// \param[in]  batch_axis     The batch axis index of data tensor.
            /// \param[in]  default_value  The default value for masked elements.
            ///
            /// \return     The masked node.
            ///
            std::shared_ptr<Node> get_masked_node(const std::shared_ptr<Node>& data,
                                                  std::int32_t time_step,
                                                  std::size_t batch_axis = 0,
                                                  const std::shared_ptr<Node>& default_value = {
                                                      nullptr})
            {
                std::shared_ptr<Node> mask_value = default_value;
                // Create zero mask value node.
                if (!mask_value)
                {
                    mask_value = op::Constant::create(
                        data->get_element_type(),
                        data->get_shape(),
                        std::vector<float>(shape_size(data->get_shape()), 0.f));
                }

                // Create predicate nodes. The condition is whether current time step value
                // is greater than sequence length for respective batch inputs.
                std::shared_ptr<Node> curr_time_step_node = op::Constant::create(
                    element::i32,
                    data->get_shape(),
                    std::vector<std::int32_t>(shape_size(data->get_shape()), time_step));

                std::shared_ptr<Node> batch_seq_length =
                    op::legacy_style_broadcast_for_binary_operation(
                        curr_time_step_node, m_seq_lengths, batch_axis)
                        .at(1);

                // Create mask node deciding whether or not to mask batch data.
                std::shared_ptr<Node> mask_condition =
                    std::make_shared<op::Greater>(curr_time_step_node, batch_seq_length);

                // Select values depnding on mask_condition.
                // Select(<condition>, <true_value>, <false_value>)
                return std::make_shared<op::Select>(mask_condition, mask_value, data);
            }

            NodeVector lstm_pass(bool is_reverse = false)
            {
                // ------ VARIABLE'S NAMES AND ACRONYM DEFINITIONS ------
                // The names used below are analogous to the one used in ONNX documentation.
                //
                // ------ INPUTS ------
                // X - The input tensor. [seq_length, batch_size, input_size]
                // W - The weight tensor. [num_directions, 4*hidden_size, input_size]
                // R - The recurrence weight tensor. [num_directions, 4*hidden_size, hidden_size]
                // B - The bias tensor for input gate. [num_directions, 8*hidden_size]
                // P - The weight tensor for peepholes. [num_directions, 3*hidde_size]
                // ------ ACRONYMS ------
                // i - input gate
                // o - output gate
                // f - forget gate
                // c - cell gate
                // t - time step (t-1 means previous time step)
                // ------ VARIABLE NAMES ------
                // H_t     - Hidden state vector at current time step.
                // C_t     - Cell state vector at current time step.
                // h_list  - The list of hidden states at all processed time steps.

                NodeVector h_list;
                std::shared_ptr<Node> X = m_X;
                std::shared_ptr<Node> W = prepare_input(m_W, is_reverse);
                std::shared_ptr<Node> R = prepare_input(m_R, is_reverse);
                std::shared_ptr<Node> B = prepare_input(m_B, is_reverse);
                std::shared_ptr<Node> P = prepare_input(m_P, is_reverse);
                std::shared_ptr<Node> H_t = prepare_input(m_initial_h, is_reverse);
                std::shared_ptr<Node> C_t = prepare_input(m_initial_c, is_reverse);

                if (is_reverse)
                {
                    X = std::make_shared<op::ReverseSequence>(
                        X, m_seq_lengths, 1 /*batch_axis*/, 0 /*seq_axis*/);
                }

                NodeVector in_seqs = builder::split(X, X->get_shape().at(0));

                for (auto& in_x : in_seqs)
                {
                    // remove first empty dim, after above split.
                    in_x = onnx_import::reshape::squeeze(in_x);
                }

                std::int32_t time_step{1};
                for (const auto& in_x : in_seqs)
                {
                    std::shared_ptr<Node> lstm_cell =
                        std::make_shared<op::LSTMCell>(in_x,
                                                       W,
                                                       R,
                                                       H_t,
                                                       C_t,
                                                       m_hidden_size,
                                                       B,
                                                       P,
                                                       m_activations,
                                                       m_activations_alpha,
                                                       m_activations_beta,
                                                       m_clip_threshold,
                                                       m_input_forget);

                    std::shared_ptr<Node> H = get_output_element(lstm_cell, 0);
                    std::shared_ptr<Node> C = get_output_element(lstm_cell, 1);

                    // Expand tensors with empty outermost dim, so we can later concatenate
                    // them.
                    // Mask hidden state tensor in order to handle mixed sequence lengths.
                    // This results in zeroing out values in batches with sequence shorter
                    // than current time_step.
                    h_list.push_back(
                        get_masked_node(onnx_import::reshape::expand_dims(H), time_step, 1));
                    // Reference implementation in ONNX Runtime doesn't mask values of Y_h
                    // and Y_c outputs, thus here we make sure that only appropriate batches
                    // (in respect to its sequence length) are updated. Those batches which
                    // has shorter sequences preserve the last value.
                    H_t = get_masked_node(H, time_step, 0, H_t);
                    C_t = get_masked_node(C, time_step, 0, C_t);
                    time_step++;
                }
                // The tensor that concats all the intermediate output values of the hidden.
                // It has shape [seq_length, batch_size, hidden_size]
                std::shared_ptr<Node> Y{std::make_shared<op::Concat>(h_list, 0)};

                // Get back the original order of the output data.
                if (is_reverse)
                {
                    Y = std::make_shared<op::ReverseSequence>(
                        Y, m_seq_lengths, 1 /*batch_axis*/, 0 /*seq_axis*/);
                }

                // Expand Y so that it has expected shape:
                // [seq_length, num_directions, batch_size, hidden_size]
                Y = onnx_import::reshape::expand_dims(Y, 1);

                // expand H_t and C_t so that it has expected shape:
                // [num_directions, batch_size, hidden_size]
                auto Y_h = onnx_import::reshape::expand_dims(H_t);
                auto Y_c = onnx_import::reshape::expand_dims(C_t);
                return {Y, Y_h, Y_c};
            }

            std::shared_ptr<Node> prepare_input(std::shared_ptr<Node> node, bool is_reverse)
            {
                // In bidirectional mode inputs are stacked together, so we must split them.
                std::shared_ptr<Node> tmp = node;
                if (m_direction == LSTMDirection::LSTM_DIRECTION_BIDIRECTIONAL)
                {
                    tmp = builder::split(node, 2).at(is_reverse ? 1 : 0);
                }
                // Since we have forward LSTM we can squeeze `num_directions` axis from inputs.
                return onnx_import::reshape::squeeze(tmp);
            }

            std::shared_ptr<Node> m_X;
            std::shared_ptr<Node> m_W;
            std::shared_ptr<Node> m_R;
            std::shared_ptr<Node> m_B;
            std::shared_ptr<Node> m_P;
            std::shared_ptr<Node> m_initial_h;
            std::shared_ptr<Node> m_initial_c;
            std::shared_ptr<Node> m_seq_lengths;

            const std::vector<float> m_activations_alpha;
            const std::vector<float> m_activations_beta;
            const std::vector<std::string> m_activations;
            const float m_clip_threshold;
            const LSTMDirection m_direction;
            const std::int64_t m_hidden_size;
            const bool m_input_forget;
        };
    } // namespace op
} // namespace ngraph
