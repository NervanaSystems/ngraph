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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/null_node.hpp"
#include "exceptions.hpp"
#include "lstm.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/builder/split.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/fused/lstm_cell.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/reverse_sequence.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/util/broadcasting.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace
            {
                // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ INPUT NODES PARSING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                enum class LSTMInput
                {
                    LSTM_INPUT_X,
                    LSTM_INPUT_W,
                    LSTM_INPUT_R,
                    LSTM_INPUT_B,
                    LSTM_INPUT_SEQ_LENGTHS,
                    LSTM_INPUT_INIT_H,
                    LSTM_INPUT_INIT_C,
                    LSTM_INPUT_P
                };

                struct LSTMNgInputMap
                {
                    using container_type = std::map<LSTMInput, std::shared_ptr<ngraph::Node>>;
                    using iterator = typename container_type::iterator;

                    explicit LSTMNgInputMap(const Node& node)
                    {
                        const auto& ng_inputs = node.get_ng_inputs();
                        // We have input, output, forget and cell gates
                        constexpr std::size_t gates_count{4};
                        // Peepholes add additional connections to input, output and forget gates.
                        constexpr std::size_t peepholes_count{3};

                        // ----- Mandatory inputs ------
                        // Packed input sequences. Shape: [seq_length, batch_size, input_size]
                        m_map[LSTMInput::LSTM_INPUT_X] = ng_inputs.at(0);
                        // Weight tensor for the gates.
                        // Shape: [num_directions, 4*hidden_size, input_size]
                        m_map[LSTMInput::LSTM_INPUT_W] = ng_inputs.at(1);
                        // The recurrence weight tensor.
                        // Shape: [num_directions, 4*hidden_size, hidden_size]
                        m_map[LSTMInput::LSTM_INPUT_R] = ng_inputs.at(2);

                        const std::size_t hidden_size =
                            m_map[LSTMInput::LSTM_INPUT_R]->get_shape().back();
                        const std::size_t batch_size =
                            m_map[LSTMInput::LSTM_INPUT_X]->get_shape().at(1);
                        const std::size_t num_directions =
                            m_map[LSTMInput::LSTM_INPUT_W]->get_shape().front();

                        // ------ Optional inputs ------
                        // The bias tensor for input gate. Shape [num_directions, 8*hidden_size]
                        if (ng_inputs.size() > 3 && !ng_inputs.at(3)->is_null())
                        {
                            m_map[LSTMInput::LSTM_INPUT_B] = ng_inputs.at(3);
                        }
                        else
                        {
                            m_map[LSTMInput::LSTM_INPUT_B] = ngraph::op::Constant::create(
                                element::f32,
                                Shape{num_directions, 2 * gates_count * hidden_size},
                                std::vector<float>(num_directions * 2 * gates_count * hidden_size,
                                                   0.f));
                        }
                        // The lengths of the sequences in a batch. Shape [batch_size]
                        if (ng_inputs.size() > 4 && !ng_inputs.at(4)->is_null())
                        {
                            m_map[LSTMInput::LSTM_INPUT_SEQ_LENGTHS] = ng_inputs.at(4);
                        }
                        else
                        {
                            m_map[LSTMInput::LSTM_INPUT_SEQ_LENGTHS] = ngraph::op::Constant::create(
                                element::i32,
                                Shape{batch_size},
                                std::vector<std::int32_t>(
                                    batch_size, m_map[LSTMInput::LSTM_INPUT_X]->get_shape().at(0)));
                        }
                        // The initial value of the hidden.
                        // Shape [num_directions, batch_size, hidden_size]
                        if (ng_inputs.size() > 5 && !ng_inputs.at(5)->is_null())
                        {
                            m_map[LSTMInput::LSTM_INPUT_INIT_H] = ng_inputs.at(5);
                        }
                        else
                        {
                            m_map[LSTMInput::LSTM_INPUT_INIT_H] = ngraph::op::Constant::create(
                                element::f32,
                                Shape{num_directions, batch_size, hidden_size},
                                std::vector<float>(num_directions * batch_size * hidden_size, 0.f));
                        }
                        // The initial value of the cell.
                        // Shape [num_directions, batch_size, hidden_size]
                        if (ng_inputs.size() > 6 && !ng_inputs.at(6)->is_null())
                        {
                            m_map[LSTMInput::LSTM_INPUT_INIT_C] = ng_inputs.at(6);
                        }
                        else
                        {
                            m_map[LSTMInput::LSTM_INPUT_INIT_C] = ngraph::op::Constant::create(
                                element::f32,
                                Shape{num_directions, batch_size, hidden_size},
                                std::vector<float>(num_directions * batch_size * hidden_size, 0.f));
                        }
                        // The weight tensor for peepholes. Shape [num_directions, 3*hidde_size]
                        if (ng_inputs.size() > 7 && !ng_inputs.at(7)->is_null())
                        {
                            m_map[LSTMInput::LSTM_INPUT_P] = ng_inputs.at(7);
                        }
                        else
                        {
                            m_map[LSTMInput::LSTM_INPUT_P] = ngraph::op::Constant::create(
                                element::f32,
                                Shape{num_directions, peepholes_count * hidden_size},
                                std::vector<float>(num_directions * peepholes_count * hidden_size,
                                                   0.f));
                        }
                    }

                    std::shared_ptr<ngraph::Node>& at(const LSTMInput& key)
                    {
                        return m_map.at(key);
                    }
                    container_type m_map;
                };

                // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ATTRIBUTES PARSING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                enum class LSTMDirection
                {
                    LSTM_DIRECTION_FORWARD,
                    LSTM_DIRECTION_REVERSE,
                    LSTM_DIRECTION_BIDIRECTIONAL,
                    LSTM_DIRECTION_UNKNOWN,
                };

                LSTMDirection getLSTMDirection(const std::string& s)
                {
                    if (s == "forward")
                    {
                        return LSTMDirection::LSTM_DIRECTION_FORWARD;
                    }
                    if (s == "reverse")
                    {
                        return LSTMDirection::LSTM_DIRECTION_REVERSE;
                    }
                    if (s == "bidirectional")
                    {
                        return LSTMDirection::LSTM_DIRECTION_BIDIRECTIONAL;
                    }
                    return LSTMDirection::LSTM_DIRECTION_UNKNOWN;
                }

                struct LSTMAttributes
                {
                    explicit LSTMAttributes(const Node& node)
                        : m_hidden_size{node.get_attribute_value<std::int64_t>("hidden_size")}
                        , m_clip_threshold{node.get_attribute_value<float>("clip", 0.f)}
                        , m_activations{node.get_attribute_value<std::vector<std::string>>(
                              "activations", {"sigmoid", "tanh", "tanh"})}
                        // Default values for activation functions are same as for corresponding
                        // ONNX operator.
                        , m_activation_alpha{node.get_attribute_value<std::vector<float>>(
                              "activation_alpha", std::vector<float>{})}
                        , m_activation_beta{node.get_attribute_value<std::vector<float>>(
                              "activation_beta", std::vector<float>{})}
                        , m_input_forget{static_cast<bool>(
                              node.get_attribute_value<std::int64_t>("input_forget", 0))}
                    {
                        m_clip_threshold = std::abs(m_clip_threshold);
                        std::string direction{ngraph::to_lower(
                            node.get_attribute_value<std::string>("direction", {"forward"}))};

                        ASSERT_VALID_ARGUMENT(node,
                                              getLSTMDirection(direction) !=
                                                  LSTMDirection::LSTM_DIRECTION_UNKNOWN)
                            << "Provided attribute \"direction\" value is incorrect: " << direction;
                        m_direction = getLSTMDirection(direction);
                    }

                    LSTMDirection m_direction;
                    std::int64_t m_hidden_size;
                    float m_clip_threshold;
                    std::vector<std::string> m_activations;
                    std::vector<float> m_activation_alpha;
                    std::vector<float> m_activation_beta;
                    bool m_input_forget;
                };

                class LSTMForward
                {
                public:
                    explicit LSTMForward(const std::shared_ptr<ngraph::Node>& X,
                                         const std::shared_ptr<ngraph::Node>& W,
                                         const std::shared_ptr<ngraph::Node>& R,
                                         const std::shared_ptr<ngraph::Node>& B,
                                         const std::shared_ptr<ngraph::Node>& P,
                                         const std::shared_ptr<ngraph::Node>& initial_h,
                                         const std::shared_ptr<ngraph::Node>& initial_c,
                                         const std::shared_ptr<ngraph::Node>& seq_lengths,
                                         const LSTMAttributes& attributes)
                        : m_X{X} // Since we have forward LSTM we can squeeze `num_directions` axis
                                 // from inputs.
                        , m_W(builder::squeeze(W))
                        , m_R(builder::squeeze(R))
                        , m_B(builder::squeeze(B))
                        , m_P(builder::squeeze(P))
                        , m_initial_h(builder::squeeze(initial_h))
                        , m_initial_c(builder::squeeze(initial_c))
                        , m_seq_lengths(seq_lengths)
                        , m_attributes(attributes)
                    {
                    }

                    NodeVector run(bool reverse = false)
                    {
                        // ------ VARIABLE'S NAMES AND ACRONYM DEFINITIONS ------
                        // The names used below are analogous to the one used in ONNX documentation.
                        //
                        // ------ INPUTS ------
                        // X - The input tensor. [seq_length, batch_size, input_size]
                        // W - The weight tensor. [num_directions, 4*hidden_size, input_size]
                        // R - The recurrence weight tensor. [num_directions, 4*hidden_size,
                        //                                    hidden_size]
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
                        std::shared_ptr<ngraph::Node> H_t = m_initial_h;
                        std::shared_ptr<ngraph::Node> C_t = m_initial_c;

                        if (reverse)
                        {
                            m_X = std::make_shared<ngraph::op::ReverseSequence>(
                                m_X, m_seq_lengths, 1 /*batch_axis*/, 0 /*seq_axis*/);
                        }

                        NodeVector in_seqs{};
                        if (m_X->get_shape().at(0) != 1)
                        {
                            in_seqs = ngraph::builder::split(m_X, m_X->get_shape().at(0));
                        }
                        else
                        {
                            in_seqs = NodeVector{m_X};
                        }

                        for (auto& in_x : in_seqs)
                        {
                            // remove first empty dim, after above split.
                            in_x = builder::squeeze(in_x);
                        }

                        std::int32_t time_step{1};
                        for (const auto& in_x : in_seqs)
                        {
                            std::shared_ptr<ngraph::Node> lstm_cell =
                                std::make_shared<ngraph::op::LSTMCell>(
                                    in_x,
                                    m_W,
                                    m_R,
                                    H_t,
                                    C_t,
                                    m_attributes.m_hidden_size,
                                    m_B,
                                    m_P,
                                    m_attributes.m_activations,
                                    m_attributes.m_activation_alpha,
                                    m_attributes.m_activation_beta,
                                    m_attributes.m_clip_threshold,
                                    m_attributes.m_input_forget);

                            std::shared_ptr<ngraph::Node> H = get_output_element(lstm_cell, 0);
                            std::shared_ptr<ngraph::Node> C = get_output_element(lstm_cell, 1);

                            // Expand tensors with empty outermost dim, so we can later concatenate
                            // them.
                            // Mask hidden state tensor in order to handle mixed sequence lengths.
                            // This results in zeroing out values in batches with sequence shorter
                            // than current time_step.
                            h_list.push_back(
                                get_masked_node(builder::expand_dims(H), time_step, 1));
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
                        std::shared_ptr<ngraph::Node> Y{
                            std::make_shared<ngraph::op::Concat>(h_list, 0)};

                        // Get back the original order of the output data.
                        if (reverse)
                        {
                            Y = std::make_shared<ngraph::op::ReverseSequence>(
                                Y, m_seq_lengths, 1 /*batch_axis*/, 0 /*seq_axis*/);
                        }

                        // Expand Y so that it has expected shape:
                        // [seq_length, num_directions, batch_size, hidden_size]
                        Y = builder::expand_dims(Y, 1);

                        // expand H_t and C_t so that it has expected shape:
                        // [num_directions, batch_size, hidden_size]
                        auto Y_h = builder::expand_dims(H_t);
                        auto Y_c = builder::expand_dims(C_t);
                        return {Y, Y_h, Y_c};
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
                    std::shared_ptr<ngraph::Node> get_masked_node(
                        const std::shared_ptr<ngraph::Node>& data,
                        std::int32_t time_step,
                        std::size_t batch_axis = 0,
                        const std::shared_ptr<ngraph::Node>& default_value = {nullptr})
                    {
                        std::shared_ptr<ngraph::Node> mask_value = default_value;
                        // Create zero mask value node.
                        if (!mask_value)
                        {
                            mask_value = ngraph::op::Constant::create(
                                data->get_element_type(),
                                data->get_shape(),
                                std::vector<float>(shape_size(data->get_shape()), 0.f));
                        }

                        // Create predicate nodes. The condition is whether current time step value
                        // is greater than sequence length for respective batch inputs.
                        std::shared_ptr<ngraph::Node> curr_time_step_node =
                            ngraph::op::Constant::create(
                                element::i32,
                                data->get_shape(),
                                std::vector<std::int32_t>(shape_size(data->get_shape()),
                                                          time_step));

                        std::shared_ptr<ngraph::Node> batch_seq_length =
                            ngraph::op::legacy_style_broadcast_for_binary_operation(
                                curr_time_step_node, m_seq_lengths, batch_axis)
                                .at(1);

                        // Create mask node deciding whether or not to mask batch data.
                        std::shared_ptr<ngraph::Node> mask_condition =
                            std::make_shared<ngraph::op::Greater>(curr_time_step_node,
                                                                  batch_seq_length);

                        // Select values depnding on mask_condition.
                        // Select(<condition>, <true_value>, <false_value>)
                        return std::make_shared<ngraph::op::Select>(
                            mask_condition, mask_value, data);
                    }

                    std::shared_ptr<ngraph::Node> m_X;
                    std::shared_ptr<ngraph::Node> m_W;
                    std::shared_ptr<ngraph::Node> m_R;
                    std::shared_ptr<ngraph::Node> m_B;
                    std::shared_ptr<ngraph::Node> m_P;
                    std::shared_ptr<ngraph::Node> m_initial_h;
                    std::shared_ptr<ngraph::Node> m_initial_c;
                    std::shared_ptr<ngraph::Node> m_seq_lengths;
                    const LSTMAttributes& m_attributes;
                };

            } // anonymous namespace

            namespace set_1
            {
                NodeVector lstm(const Node& node)
                {
                    LSTMNgInputMap input_map{node};
                    LSTMAttributes attributes{node};

                    NodeVector results;

                    if (attributes.m_direction == LSTMDirection::LSTM_DIRECTION_FORWARD ||
                        attributes.m_direction == LSTMDirection::LSTM_DIRECTION_REVERSE)
                    {
                        LSTMForward lstm_fwd(input_map.at(LSTMInput::LSTM_INPUT_X),
                                             input_map.at(LSTMInput::LSTM_INPUT_W),
                                             input_map.at(LSTMInput::LSTM_INPUT_R),
                                             input_map.at(LSTMInput::LSTM_INPUT_B),
                                             input_map.at(LSTMInput::LSTM_INPUT_P),
                                             input_map.at(LSTMInput::LSTM_INPUT_INIT_H),
                                             input_map.at(LSTMInput::LSTM_INPUT_INIT_C),
                                             input_map.at(LSTMInput::LSTM_INPUT_SEQ_LENGTHS),
                                             attributes);
                        results = lstm_fwd.run(
                            (attributes.m_direction == LSTMDirection::LSTM_DIRECTION_REVERSE));
                    }
                    if (attributes.m_direction == LSTMDirection::LSTM_DIRECTION_BIDIRECTIONAL)
                    {
                        // In bidirectional mode weights are stacked together, so we must split
                        // them.
                        NodeVector W{
                            ngraph::builder::split(input_map.at(LSTMInput::LSTM_INPUT_W), 2)};
                        NodeVector R{
                            ngraph::builder::split(input_map.at(LSTMInput::LSTM_INPUT_R), 2)};
                        NodeVector B{
                            ngraph::builder::split(input_map.at(LSTMInput::LSTM_INPUT_B), 2)};
                        NodeVector P{
                            ngraph::builder::split(input_map.at(LSTMInput::LSTM_INPUT_P), 2)};
                        NodeVector H{
                            ngraph::builder::split(input_map.at(LSTMInput::LSTM_INPUT_INIT_H), 2)};
                        NodeVector C{
                            ngraph::builder::split(input_map.at(LSTMInput::LSTM_INPUT_INIT_C), 2)};

                        LSTMForward lstm_fwd(input_map.at(LSTMInput::LSTM_INPUT_X),
                                             W.at(0),
                                             R.at(0),
                                             B.at(0),
                                             P.at(0),
                                             H.at(0),
                                             C.at(0),
                                             input_map.at(LSTMInput::LSTM_INPUT_SEQ_LENGTHS),
                                             attributes);
                        LSTMForward lstm_reversed(input_map.at(LSTMInput::LSTM_INPUT_X),
                                                  W.at(1),
                                                  R.at(1),
                                                  B.at(1),
                                                  P.at(1),
                                                  H.at(1),
                                                  C.at(1),
                                                  input_map.at(LSTMInput::LSTM_INPUT_SEQ_LENGTHS),
                                                  attributes);

                        NodeVector fwd_results{lstm_fwd.run()};
                        NodeVector rev_results{lstm_reversed.run(true)};

                        // Stack together respective outputs from both forward and reverse passess.
                        std::shared_ptr<ngraph::Node> Y{std::make_shared<ngraph::op::Concat>(
                            NodeVector{fwd_results.at(0), rev_results.at(0)}, 1)};
                        std::shared_ptr<ngraph::Node> Y_h{std::make_shared<ngraph::op::Concat>(
                            NodeVector{fwd_results.at(1), rev_results.at(1)}, 0)};
                        std::shared_ptr<ngraph::Node> Y_c{std::make_shared<ngraph::op::Concat>(
                            NodeVector{fwd_results.at(2), rev_results.at(2)}, 0)};
                        results = NodeVector{Y, Y_h, Y_c};
                    }

                    return results;
                }
            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
