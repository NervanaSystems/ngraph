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
#include "ngraph/op/concat.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

#include "exceptions.hpp"
#include "lstm.hpp"
#include "utils/arithmetic_operators.hpp"
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
                using NgraphNodePtr = std::shared_ptr<ngraph::Node>;

                // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ACTIVATION FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                NgraphNodePtr Sigmoid(const NgraphNodePtr& arg)
                {
                    return std::make_shared<ngraph::op::Sigmoid>(arg);
                }

                NgraphNodePtr Tanh(const NgraphNodePtr& arg)
                {
                    return std::make_shared<ngraph::op::Tanh>(arg);
                }

                // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ INPUT NODES PARSING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                struct LSTMNgInputMap
                {
                    using iterator = std::map<std::string, NgraphNodePtr>::iterator;

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
                        const std::size_t n_dirs = m_map["W"]->get_shape().front();

                        // ------ Optional inputs ------
                        // The bias tensor for input gate. Shape [num_directions, 8*hidden_size]
                        if (ng_inputs.size() >= 4)
                        {
                            m_map["B"] = ng_inputs.at(3);
                        }
                        else
                        {
                            m_map["B"] = common::make_constant_node<float>(
                                element::f32, {n_dirs, 2 * gates_count * hidden_size}, {0.f});
                        }
                        // The lengths of the sequences in a batch. Shape [batch_size]
                        if (ng_inputs.size() >= 5)
                        {
                            m_map["seq_lengths"] = ng_inputs.at(4);
                        }
                        // The initial value of the hidden. Shape [num_directions, batch_size, hidden_size]
                        if (ng_inputs.size() >= 6)
                        {
                            m_map["init_H"] = ng_inputs.at(5);
                        }
                        else
                        {
                            m_map["init_H"] = common::make_constant_node<float>(
                                element::f32, {n_dirs, batch_size, hidden_size}, {0.f});
                        }
                        // The initial value of the cell. Shape [num_directions, batch_size, hidden_size]
                        if (ng_inputs.size() >= 7)
                        {
                            m_map["init_C"] = ng_inputs.at(6);
                        }
                        else
                        {
                            m_map["init_C"] = common::make_constant_node<float>(
                                element::f32, {n_dirs, batch_size, hidden_size}, {0.f});
                        }
                        // The weight tensor for peepholes. Shape [num_directions, 3*hidde_size]
                        if (ng_inputs.size() >= 8)
                        {
                            m_map["P"] = ng_inputs.at(7);
                        }
                        else
                        {
                            m_map["P"] = common::make_constant_node<float>(
                                element::f32, {n_dirs, peepholes_count * hidden_size}, {0.f});
                        }
                    }

                    NgraphNodePtr& operator[](const std::string& key) { return m_map[key]; }
                    iterator begin() { return m_map.begin(); }
                    iterator end() { return m_map.end(); }
                    std::map<std::string, NgraphNodePtr> m_map;
                };

                // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ATTRIBUTES PARSING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                enum class LSTMDirection
                {
                    LSTM_DIRECTION_FORWARD,
                    LSTM_DIRECTION_REVERSE,
                    LSTM_DIRECTION_BIDIRECTIONAL
                };

                using ActivationFunc = std::function<NgraphNodePtr(const NgraphNodePtr&)>;
                using ActivationFuncsMap = std::unordered_map<std::string, ActivationFunc>;

                struct LSTMAttributes
                {
                    explicit LSTMAttributes(const Node& node)
                    {
                        // ---- Required -----
                        m_hidden_size = node.get_attribute_value<std::int64_t>("hidden_size");

                        // ---- Optional -----
                        m_activation_alpha =
                            node.get_attribute_value<std::vector<float>>("activation_alpha", {});
                        m_activation_beta =
                            node.get_attribute_value<std::vector<float>>("activation_beta", {});

                        // If absent - no clipping.
                        m_clip = node.get_attribute_value<float>(
                            "clip", {std::numeric_limits<float>::max()});
                        ASSERT_IS_SUPPORTED(node, (m_clip == std::numeric_limits<float>::max()))
                            << "Currently clipping is not supported.";

                        std::string direction =
                            node.get_attribute_value<std::string>("direction", "forward");
                        ASSERT_IS_SUPPORTED(node, (direction == "forward"))
                            << "Currently only forward mode is supported.";

                        m_input_forget = static_cast<bool>(
                            node.get_attribute_value<std::int64_t>("input_forget", 0));
                        ASSERT_IS_SUPPORTED(node, (m_input_forget == 0))
                            << "Coupling input and forget gates is currently not supported.";

                        // Register available activation functions.
                        m_atcivation_funcs.emplace("Sigmoid",
                                                   std::bind(Sigmoid, std::placeholders::_1));
                        m_atcivation_funcs.emplace("Tanh", std::bind(Tanh, std::placeholders::_1));
                    }

                    std::vector<float> m_activation_alpha{};
                    std::vector<float> m_activation_beta{};
                    ActivationFuncsMap m_atcivation_funcs{};
                    float m_clip{std::numeric_limits<float>::max()};
                    LSTMDirection m_direction{LSTMDirection::LSTM_DIRECTION_FORWARD};
                    std::int64_t m_hidden_size{0};
                    bool m_input_forget{false};
                };

                // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ LSTM NODE CLASS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                class LSTMNode
                {
                public:
                    explicit LSTMNode(const Node& node)
                        : m_input_map{node}
                        , m_attributes{node}
                        , m_f{m_attributes.m_atcivation_funcs["Sigmoid"]}
                        , m_g{m_attributes.m_atcivation_funcs["Tanh"]}
                        , m_h{m_attributes.m_atcivation_funcs["Tanh"]}
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
                    ~LSTMNode(){};

                    NodeVector run()
                    {
                        NodeVector p_iof = reshape::split(m_input_map["P"], 3);
                        NgraphNodePtr p_i = p_iof.at(0);
                        NgraphNodePtr p_o = p_iof.at(1);
                        NgraphNodePtr p_f = p_iof.at(2);
                        NgraphNodePtr H_t = m_input_map["init_H"];
                        NgraphNodePtr C_t = m_input_map["init_C"];
                        NodeVector h_list;

                        NodeVector b_W_R = reshape::split(m_input_map["B"], 2);
                        NgraphNodePtr bias = b_W_R.at(0) + b_W_R.at(1);
                        NodeVector in_seqs =
                            reshape::split(m_input_map["X"], m_input_map["X"]->get_shape().at(0));
                        for (auto& in_x : in_seqs)
                        {
                            // remove first empty dim, after above split.
                            in_x = reshape::squeeze(in_x);
                        }

                        for (const auto& in_x : in_seqs)
                        {
                            auto Xt_W = std::make_shared<ngraph::op::Dot>(
                                in_x, reshape::transpose(m_input_map["W"]));
                            auto Ht_W = std::make_shared<ngraph::op::Dot>(
                                H_t, reshape::transpose(m_input_map["R"]));
                            auto gates = Xt_W + Ht_W + bias;

                            NodeVector split_gates = reshape::split(gates, 4, -1);
                            auto i = split_gates.at(0);
                            auto o = split_gates.at(1);
                            auto f = split_gates.at(2);
                            auto c = split_gates.at(3);

                            i = m_f(i + p_i * C_t);
                            f = m_f(f + p_f * C_t);
                            auto C = f * C_t + i * m_g(c);
                            o = m_f(o + p_o * C);
                            auto H = o * m_h(C);
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
                        NgraphNodePtr Y{std::make_shared<ngraph::op::Concat>(exp_h_list, 0)};

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

                    const ActivationFunc& m_f;
                    const ActivationFunc& m_g;
                    const ActivationFunc& m_h;

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
