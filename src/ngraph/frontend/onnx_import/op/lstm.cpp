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
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/null_node.hpp"
#include "exceptions.hpp"
#include "lstm.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/builder/make_constant.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/builder/split.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/op/util/broadcasting.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/util.hpp"
#include "utils/reshape.hpp"
#include "utils/rnn/activation_functions.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace
            {
                std::shared_ptr<ngraph::Node> add(const std::shared_ptr<ngraph::Node>& lhs,
                                                  const std::shared_ptr<ngraph::Node>& rhs)
                {
                    auto args = ngraph::op::numpy_style_broadcast({lhs, rhs});
                    return {std::make_shared<ngraph::op::Add>(args.at(0), args.at(1))};
                }

                std::shared_ptr<ngraph::Node> sub(const std::shared_ptr<ngraph::Node>& lhs,
                                                  const std::shared_ptr<ngraph::Node>& rhs)
                {
                    auto args = ngraph::op::numpy_style_broadcast({lhs, rhs});
                    return {std::make_shared<ngraph::op::Subtract>(args.at(0), args.at(1))};
                }

                std::shared_ptr<ngraph::Node> mul(const std::shared_ptr<ngraph::Node>& lhs,
                                                  const std::shared_ptr<ngraph::Node>& rhs)
                {
                    auto args = ngraph::op::numpy_style_broadcast({lhs, rhs});
                    return {std::make_shared<ngraph::op::Multiply>(args.at(0), args.at(1))};
                }

                std::shared_ptr<ngraph::Node> clip(const std::shared_ptr<ngraph::Node>& data,
                                                   float threshold)
                {
                    if (threshold == 0.f)
                    {
                        return data;
                    }

                    float min_val = -threshold;
                    float max_val = threshold;
                    std::size_t size = ngraph::shape_size(data->get_shape());
                    const std::shared_ptr<ngraph::Node> min_val_node =
                        ngraph::op::Constant::create(data->get_element_type(),
                                                     data->get_shape(),
                                                     std::vector<float>(size, min_val));
                    const std::shared_ptr<ngraph::Node> max_val_node =
                        ngraph::op::Constant::create(data->get_element_type(),
                                                     data->get_shape(),
                                                     std::vector<float>(size, max_val));

                    return std::make_shared<ngraph::op::Minimum>(
                        max_val_node, std::make_shared<ngraph::op::Maximum>(data, min_val_node));
                }

                // Modify input vector in-place and return reference to modified vector.
                std::vector<std::string>& to_lower_case(std::vector<std::string>&& vs)
                {
                    std::transform(std::begin(vs),
                                   std::end(vs),
                                   std::begin(vs),
                                   [](std::string& s) { return ngraph::to_lower(s); });
                    return vs;
                }

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
                        // Weight tensor for the gates. Shape: [num_directions, 4*hidden_size, input_size]
                        m_map[LSTMInput::LSTM_INPUT_W] = ng_inputs.at(1);
                        // The recurrence weight tensor. Shape: [num_directions, 4*hidden_size, hidden_size]
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
                            m_map[LSTMInput::LSTM_INPUT_B] = ngraph::builder::make_constant<float>(
                                element::f32, {num_directions, 2 * gates_count * hidden_size}, 0.f);
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
                        // The initial value of the hidden. Shape [num_directions, batch_size, hidden_size]
                        if (ng_inputs.size() > 5 && !ng_inputs.at(5)->is_null())
                        {
                            m_map[LSTMInput::LSTM_INPUT_INIT_H] = ng_inputs.at(5);
                        }
                        else
                        {
                            m_map[LSTMInput::LSTM_INPUT_INIT_H] =
                                ngraph::builder::make_constant<float>(
                                    element::f32, {num_directions, batch_size, hidden_size}, 0.f);
                        }
                        // The initial value of the cell. Shape [num_directions, batch_size, hidden_size]
                        if (ng_inputs.size() > 6 && !ng_inputs.at(6)->is_null())
                        {
                            m_map[LSTMInput::LSTM_INPUT_INIT_C] = ng_inputs.at(6);
                        }
                        else
                        {
                            m_map[LSTMInput::LSTM_INPUT_INIT_C] =
                                ngraph::builder::make_constant<float>(
                                    element::f32, {num_directions, batch_size, hidden_size}, 0.f);
                        }
                        // The weight tensor for peepholes. Shape [num_directions, 3*hidde_size]
                        if (ng_inputs.size() > 7 && !ng_inputs.at(7)->is_null())
                        {
                            m_map[LSTMInput::LSTM_INPUT_P] = ng_inputs.at(7);
                        }
                        else
                        {
                            m_map[LSTMInput::LSTM_INPUT_P] = ngraph::builder::make_constant<float>(
                                element::f32, {num_directions, peepholes_count * hidden_size}, 0.f);
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
                        , m_activations{to_lower_case(
                              node.get_attribute_value<std::vector<std::string>>(
                                  "activations", {"sigmoid", "tanh", "tanh"}))}
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
                    explicit LSTMForward(std::shared_ptr<ngraph::Node> X,
                                         std::shared_ptr<ngraph::Node> W,
                                         std::shared_ptr<ngraph::Node> R,
                                         std::shared_ptr<ngraph::Node> B,
                                         std::shared_ptr<ngraph::Node> P,
                                         std::shared_ptr<ngraph::Node> initial_h,
                                         std::shared_ptr<ngraph::Node> initial_c,
                                         std::shared_ptr<ngraph::Node> seq_lengths,
                                         rnn::ActivationFunction activation_f,
                                         rnn::ActivationFunction activation_g,
                                         rnn::ActivationFunction activation_h,
                                         bool input_forget = false,
                                         float clip_threshold = 0.f)
                        : m_X{X}
                        // Since we have forward LSTM we can squeeze `num_directions` axis from inputs.
                        , m_W{reshape::squeeze(W)}
                        , m_R{reshape::squeeze(R)}
                        , m_B{reshape::squeeze(B)}
                        , m_P{reshape::squeeze(P)}
                        , m_initial_h{reshape::squeeze(initial_h)}
                        , m_initial_c{reshape::squeeze(initial_c)}
                        , m_seq_lengths{seq_lengths}
                        , m_activation_f{activation_f}
                        , m_activation_g{activation_g}
                        , m_activation_h{activation_h}
                        , m_input_forget{input_forget}
                        , m_clip_threshold{clip_threshold}
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
                        // R - The recurrence weight tensor. [num_directions, 4*hidden_size, hidden_size]
                        // B - The bias tensor for input gate. [num_directions, 8*hidden_size]
                        // P - The weight tensor forr peepholes. [num_directions, 3*hidde_size]
                        // ------ ACRONYMS ------
                        // i - input gate
                        // o - output gate
                        // f - forget gate
                        // c - cell gate
                        // t - time step (t-1 means previous time step)
                        // ------ VARIABLE NAMES ------
                        // W       - W parameter weight matrix for input, output, forget, and
                        //           cell gates.
                        // R       - R recurrence weight matrix for input, output, forget, and
                        //           cell gates.
                        // Wb      - W bias vectors for input, output, forget, and cell gates.
                        // Rb      - R bias vectors for input, output, forget, and cell gates.
                        // b_W_R   - Bias vectors for input, output, forget, and cell gates.
                        //           Concatenation of `[Wb, Rb]`.
                        // p_[iof] - P peephole weight vector for respectively: input, output,
                        //           and forget gates.
                        // H_t     - Hidden state vector at current time step.
                        // C_t     - Cell state vector at current time step.
                        // h_list  - The list of hidden states at all processed time steps.
                        //
                        // Xt_W    - Input sequence multiplied by weights tensor at current time
                        //           step.
                        // Ht_R    - Hidden state multiplied by weights tensor at current time step.

                        NodeVector p_iof = ngraph::builder::split(m_P, 3);
                        const auto& p_i = p_iof.at(0);
                        const auto& p_o = p_iof.at(1);
                        const auto& p_f = p_iof.at(2);
                        NodeVector h_list;

                        NodeVector b_W_R = ngraph::builder::split(m_B, 2);
                        std::shared_ptr<ngraph::Node> bias = b_W_R.at(0) + b_W_R.at(1);
                        std::shared_ptr<ngraph::Node> H_t = m_initial_h;
                        std::shared_ptr<ngraph::Node> C_t = m_initial_c;

                        if (reverse)
                        {
                            m_X = std::make_shared<ngraph::op::Reverse>(m_X, AxisSet{0});
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
                            in_x = reshape::squeeze(in_x);
                        }

                        std::int32_t time_step{1};
                        for (const auto& in_x : in_seqs)
                        {
                            // (.) - Denotes element-wise multiplication.
                            // *   - Denotes dot product.

                            // Xt*(W^T) -- for [iofc] gates.
                            auto Xt_W = std::make_shared<ngraph::op::Dot>(
                                in_x, ngraph::builder::transpose(m_W));
                            // Ht-1*(R^T)  -- for [iofc] gates.
                            auto Ht_R = std::make_shared<ngraph::op::Dot>(
                                H_t, ngraph::builder::transpose(m_R));
                            // Xt*(W^T) + Ht-1*(R^T) + Wb + Rb  -- for [iofc] gates.
                            auto gates = add(Xt_W, add(Ht_R, bias));

                            NodeVector split_gates = ngraph::builder::split(gates, 4, -1);
                            auto i = split_gates.at(0);
                            auto o = split_gates.at(1);
                            auto f = split_gates.at(2);
                            auto c = split_gates.at(3);

                            // f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
                            i = m_activation_f(clip(add(i, mul(p_i, C_t)), m_clip_threshold));
                            if (m_input_forget)
                            {
                                // Couple input with forget gate: 1 - i
                                f = sub(ngraph::op::Constant::create(
                                            i->get_element_type(),
                                            i->get_shape(),
                                            std::vector<float>(shape_size(i->get_shape()), 1.f)),
                                        i);
                            }
                            else
                            {
                                // f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
                                f = m_activation_f(clip(add(f, mul(p_f, C_t)), m_clip_threshold));
                            }
                            // ft (.) Ct-1 + it (.) ct
                            auto C =
                                add(mul(f, C_t), mul(i, m_activation_g(clip(c, m_clip_threshold))));
                            // f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
                            o = m_activation_f(clip(add(o, mul(p_o, C)), m_clip_threshold));
                            // ot (.) h(Ct)
                            auto H = mul(o, m_activation_h(C));

                            // Expand tensors with empty outermost dim, so we can later concatenate
                            // them.
                            // Mask hidden state tensor in order to handle mixed sequence lengths.
                            // This results in zeroing out values in batches with sequence shorter
                            // than current time_step.
                            h_list.push_back(
                                get_masked_node(reshape::expand_dims(H), time_step, 1));
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
                            Y = std::make_shared<ngraph::op::Reverse>(Y, AxisSet{0});
                        }

                        // Expand Y so that it has expected shape:
                        // [seq_length, num_directions, batch_size, hidden_size]
                        Y = reshape::expand_dims(Y, 1);

                        // expand H_t and C_t so that it has expected shape:
                        // [num_directions, batch_size, hidden_size]
                        auto Y_h = reshape::expand_dims(H_t);
                        auto Y_c = reshape::expand_dims(C_t);
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
                    rnn::ActivationFunction m_activation_f;
                    rnn::ActivationFunction m_activation_g;
                    rnn::ActivationFunction m_activation_h;
                    // For coupling input and forget gates.
                    bool m_input_forget;
                    // For clipping cell input in the range [-clip_threshold, clip_threshold].
                    float m_clip_threshold;
                };

                rnn::ActivationFunction get_activation_function(const LSTMAttributes& attributes,
                                                                std::size_t idx)
                {
                    rnn::ActivationFunction afunc =
                        rnn::get_activation_func_by_name(attributes.m_activations.at(idx));

                    // Set activation functions parameters (if any)
                    if (attributes.m_activation_alpha.size() > idx)
                    {
                        afunc.set_alpha(attributes.m_activation_alpha.at(idx));
                    }
                    if (attributes.m_activation_beta.size() > idx)
                    {
                        afunc.set_beta(attributes.m_activation_beta.at(idx));
                    }

                    return afunc;
                }

            } // anonymous namespace

            namespace set_1
            {
                NodeVector lstm(const Node& node)
                {
                    LSTMNgInputMap input_map{node};
                    LSTMAttributes attributes{node};

                    // Get activation functions.
                    const rnn::ActivationFunction& activation_f =
                        get_activation_function(attributes, 0);
                    const rnn::ActivationFunction& activation_g =
                        get_activation_function(attributes, 1);
                    const rnn::ActivationFunction& activation_h =
                        get_activation_function(attributes, 2);

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
                                             activation_f,
                                             activation_g,
                                             activation_h,
                                             attributes.m_input_forget,
                                             attributes.m_clip_threshold);
                        results = lstm_fwd.run(
                            (attributes.m_direction == LSTMDirection::LSTM_DIRECTION_REVERSE));
                    }
                    if (attributes.m_direction == LSTMDirection::LSTM_DIRECTION_BIDIRECTIONAL)
                    {
                        // In bidirectional mode weights are stacked together, so we must split them.
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
                                             activation_f,
                                             activation_g,
                                             activation_h,
                                             attributes.m_input_forget,
                                             attributes.m_clip_threshold);
                        LSTMForward lstm_reversed(input_map.at(LSTMInput::LSTM_INPUT_X),
                                                  W.at(1),
                                                  R.at(1),
                                                  B.at(1),
                                                  P.at(1),
                                                  H.at(1),
                                                  C.at(1),
                                                  input_map.at(LSTMInput::LSTM_INPUT_SEQ_LENGTHS),
                                                  activation_f,
                                                  activation_g,
                                                  activation_h,
                                                  attributes.m_input_forget,
                                                  attributes.m_clip_threshold);

                        NodeVector fwd_results{lstm_fwd.run()};
                        NodeVector rev_results{lstm_fwd.run(true)};

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

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
