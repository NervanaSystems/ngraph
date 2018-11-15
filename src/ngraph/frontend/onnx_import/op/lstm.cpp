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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ngraph/node.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

#include "exceptions.hpp"
#include "lstm.hpp"
#include "utils/broadcasting.hpp"
#include "utils/common.hpp"
#include "utils/reshape.hpp"

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
                    auto args = numpy_style_broadcast_for_binary_operation(lhs, rhs);
                    return {std::make_shared<ngraph::op::Add>(args.at(0), args.at(1))};
                }

                std::shared_ptr<ngraph::Node> mul(const std::shared_ptr<ngraph::Node>& lhs,
                                                  const std::shared_ptr<ngraph::Node>& rhs)
                {
                    auto args = numpy_style_broadcast_for_binary_operation(lhs, rhs);
                    return {std::make_shared<ngraph::op::Multiply>(args.at(0), args.at(1))};
                }

                // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ACTIVATION FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                std::shared_ptr<ngraph::Node> Sigmoid(const std::shared_ptr<ngraph::Node>& arg)
                {
                    return std::make_shared<ngraph::op::Sigmoid>(arg);
                }

                std::shared_ptr<ngraph::Node> Tanh(const std::shared_ptr<ngraph::Node>& arg)
                {
                    return std::make_shared<ngraph::op::Tanh>(arg);
                }

                // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ INPUT NODES PARSING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                struct LSTMNgInputMap
                {
                    using iterator = std::map<std::string, std::shared_ptr<ngraph::Node>>::iterator;

                    explicit LSTMNgInputMap(const Node& node)
                    {
                        const auto& ng_inputs = node.get_ng_inputs();
                        const std::size_t gates_count{4};
                        const std::size_t peepholes_count{3};

                        // ----- Mandatory inputs ------
                        // Packed input sequences. Shape: [seq_length, batch_size, input_size]
                        m_map["X"] = ng_inputs.at(0);
                        // Weight tensor for the gates. Shape: [num_directions, 4*hidden_size, input_size]
                        m_map["W"] = ng_inputs.at(1);
                        // The recurrence weight tensor. Shape: [num_directions, 4*hidden_size, hidden_size]
                        m_map["R"] = ng_inputs.at(2);

                        const std::size_t hidden_size = m_map["R"]->get_shape().back();
                        const std::size_t batch_size = m_map["X"]->get_shape().at(1);
                        const std::size_t num_directions = m_map["W"]->get_shape().front();

                        // ------ Optional inputs ------
                        // The bias tensor for input gate. Shape [num_directions, 8*hidden_size]
                        if (ng_inputs.size() >= 4)
                        {
                            m_map["B"] = ng_inputs.at(3);
                        }
                        else
                        {
                            m_map["B"] = common::make_constant_node<float>(
                                element::f32,
                                {num_directions, 2 * gates_count * hidden_size},
                                {0.f});
                        }
                        // The lengths of the sequences in a batch. Shape [batch_size]
                        if (ng_inputs.size() >= 5)
                        {
                            m_map["seq_lengths"] = ng_inputs.at(4);
                        }
                        else
                        {
                            m_map["seq_lengths"] = common::make_constant_node<std::int32_t>(
                                element::i32,
                                {batch_size},
                                {static_cast<std::int32_t>(m_map["X"]->get_shape().at(0))});
                        }
                        // The initial value of the hidden. Shape [num_directions, batch_size, hidden_size]
                        if (ng_inputs.size() >= 6)
                        {
                            m_map["init_H"] = ng_inputs.at(5);
                        }
                        else
                        {
                            m_map["init_H"] = common::make_constant_node<float>(
                                element::f32, {num_directions, batch_size, hidden_size}, {0.f});
                        }
                        // The initial value of the cell. Shape [num_directions, batch_size, hidden_size]
                        if (ng_inputs.size() >= 7)
                        {
                            m_map["init_C"] = ng_inputs.at(6);
                        }
                        else
                        {
                            m_map["init_C"] = common::make_constant_node<float>(
                                element::f32, {num_directions, batch_size, hidden_size}, {0.f});
                        }
                        // The weight tensor for peepholes. Shape [num_directions, 3*hidde_size]
                        if (ng_inputs.size() >= 8)
                        {
                            m_map["P"] = ng_inputs.at(7);
                        }
                        else
                        {
                            m_map["P"] = common::make_constant_node<float>(
                                element::f32,
                                {num_directions, peepholes_count * hidden_size},
                                {0.f});
                        }
                    }

                    std::shared_ptr<ngraph::Node>& operator[](const std::string& key) { return m_map[key]; }
                    iterator begin() { return m_map.begin(); }
                    iterator end() { return m_map.end(); }
                    std::map<std::string, std::shared_ptr<ngraph::Node>> m_map;
                };

                // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ATTRIBUTES PARSING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                enum class LSTMDirection
                {
                    LSTM_DIRECTION_FORWARD,
                    LSTM_DIRECTION_REVERSE,
                    LSTM_DIRECTION_BIDIRECTIONAL
                };

                using ActivationFunc = std::function<std::shared_ptr<ngraph::Node>(
                    const std::shared_ptr<ngraph::Node>&)>;
                using ActivationFuncsMap = std::unordered_map<std::string, ActivationFunc>;

                struct LSTMAttributes
                {
                    explicit LSTMAttributes(const Node& node)
                    {
                        // ---- Required -----
                        m_hidden_size = node.get_attribute_value<std::int64_t>("hidden_size");

                        // Register available activation functions.
                        m_activation_funcs.emplace("Sigmoid",
                                                   std::bind(Sigmoid, std::placeholders::_1));
                        m_activation_funcs.emplace("Tanh", std::bind(Tanh, std::placeholders::_1));
                    }

                    ActivationFuncsMap m_activation_funcs{};
                    // Currently only LSTM_DIRECTION_FORWARD is supported.
                    LSTMDirection m_direction{LSTMDirection::LSTM_DIRECTION_FORWARD};
                    std::int64_t m_hidden_size{0};
                };

                // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LSTM NODE CLASS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                class LSTMNode
                {
                public:
                    explicit LSTMNode(const Node& node)
                        : m_input_map{node}
                        , m_attributes{node}
                        , m_activ_func_f{m_attributes.m_activation_funcs["Sigmoid"]}
                        , m_activ_func_g{m_attributes.m_activation_funcs["Tanh"]}
                        , m_activ_func_h{m_attributes.m_activation_funcs["Tanh"]}
                    {
                        if (m_attributes.m_direction == LSTMDirection::LSTM_DIRECTION_FORWARD)
                        {
                            // Since we have forward LSTM we can squeeze `num_directions` axis from inputs.
                            for (auto& ng_in : m_input_map)
                            {
                                if (ng_in.first != "X" && ng_in.first != "seq_lengths")
                                {
                                    ASSERT_VALID_ARGUMENT(node,
                                                          ng_in.second->get_shape().at(0) == 1)
                                        << "Input: { " << ng_in.first
                                        << " } first axis has size different "
                                           "from 1, while direction attribute set to 'forward'.";
                                    ng_in.second = reshape::squeeze(ng_in.second);
                                }
                            }
                        }
                    }

                    ~LSTMNode() {}
                    NodeVector run()
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

                        NodeVector p_iof = reshape::split(m_input_map["P"], 3);
                        std::shared_ptr<ngraph::Node> p_i = p_iof.at(0);
                        std::shared_ptr<ngraph::Node> p_o = p_iof.at(1);
                        std::shared_ptr<ngraph::Node> p_f = p_iof.at(2);
                        std::shared_ptr<ngraph::Node> H_t = m_input_map["init_H"];
                        std::shared_ptr<ngraph::Node> C_t = m_input_map["init_C"];
                        NodeVector h_list;

                        NodeVector b_W_R = reshape::split(m_input_map["B"], 2);
                        std::shared_ptr<ngraph::Node> bias = b_W_R.at(0) + b_W_R.at(1);
                        NodeVector in_seqs =
                            reshape::split(m_input_map["X"], m_input_map["X"]->get_shape().at(0));
                        for (auto& in_x : in_seqs)
                        {
                            // remove first empty dim, after above split.
                            in_x = reshape::squeeze(in_x);
                        }

                        for (const auto& in_x : in_seqs)
                        {
                            // (.) - Denotes element-wise multiplication.
                            // *   - Denotes dot product.

                            // Xt*(W^T) -- for [iofc] gates.
                            auto Xt_W = std::make_shared<ngraph::op::Dot>(
                                in_x, reshape::transpose(m_input_map["W"]));
                            // Ht-1*(R^T)  -- for [iofc] gates.
                            auto Ht_R = std::make_shared<ngraph::op::Dot>(
                                H_t, reshape::transpose(m_input_map["R"]));
                            // Xt*(W^T) + Ht-1*(R^T) + Wb + Rb  -- for [iofc] gates.
                            auto gates = add(Xt_W, add(Ht_R, bias));

                            NodeVector split_gates = reshape::split(gates, 4, -1);
                            // Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi
                            auto i = split_gates.at(0);
                            // Xt*(Wo^T) + Ht-1*(Ro^T) + Wbo + Rbo
                            auto o = split_gates.at(1);
                            // Xt*(Wf^T) + Ht-1*(Rf^T) + Wbf + Rbf
                            auto f = split_gates.at(2);
                            // Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc
                            auto c = split_gates.at(3);

                            // f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
                            i = m_activ_func_f(add(i, mul(p_i, C_t)));
                            // f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
                            f = m_activ_func_f(add(f, mul(p_f, C_t)));
                            // ft (.) Ct-1 + it (.) ct
                            auto C = add(mul(f, C_t), mul(i, m_activ_func_g(c)));
                            // f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
                            o = m_activ_func_f(add(o, mul(p_o, C)));
                            // ot (.) h(Ct)
                            auto H = mul(o, m_activ_func_h(C));
                            h_list.push_back(H);
                            H_t = H;
                            C_t = C;
                        }
                        // The tensor that concats all the intermediate output values of the hidden.
                        // It has shape [seq_length, batch_size, hidden_size]
                        NodeVector exp_h_list;
                        for (const auto& ht : h_list)
                        {
                            exp_h_list.push_back(reshape::add_empty_axes(ht));
                        }
                        std::shared_ptr<ngraph::Node> Y{
                            std::make_shared<ngraph::op::Concat>(exp_h_list, 0)};

                        // Expand Y so that it has expected shape:
                        // [seq_length, num_directions, batch_size, hidden_size]
                        if (m_attributes.m_direction == LSTMDirection::LSTM_DIRECTION_FORWARD)
                        {
                            Shape shape{Y->get_shape()};
                            shape.insert(std::next(std::begin(shape)), 1);
                            Y = std::make_shared<ngraph::op::Reshape>(
                                Y, reshape::get_default_axis_vector(Y->get_shape().size()), shape);
                        }
                        return {Y, exp_h_list.back()};
                    }

                private:
                    LSTMNgInputMap m_input_map;
                    LSTMAttributes m_attributes;

                    const ActivationFunc& m_activ_func_f;
                    const ActivationFunc& m_activ_func_g;
                    const ActivationFunc& m_activ_func_h;

                    // input, output, cell, forget
                    const std::size_t m_gates_count{4};
                    // input, output, forget
                    const std::size_t m_peepholes_count{3};
                };

            } // anonymous namespace

            namespace set_1
            {
                NodeVector lstm(const Node& node)
                {
                    LSTMNode lstm{node};
                    return lstm.run();
                }
            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
