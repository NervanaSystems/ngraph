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

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ngraph/node.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/fused/lstm_cell.hpp"
#include "ngraph/op/util/fused_op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            ///
            /// \brief      Class for lstm sequence node.
            ///
            /// \note       It follows notation and equations defined as in ONNX standard:
            ///             https://github.com/onnx/onnx/blob/master/docs/Operators.md#LSTM
            ///
            /// \sa         LSTMCell, RNNCell, GRUCell
            ///
            ///
            class NGRAPH_API LSTMSequence : public util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"LSTMSequence", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                LSTMSequence() = default;

                enum class direction
                {
                    FORWARD,
                    REVERSE,
                    BIDIRECTIONAL
                };

                explicit LSTMSequence(const Output<Node>& X,
                                      const Output<Node>& initial_hidden_state,
                                      const Output<Node>& initial_cell_state,
                                      const Output<Node>& sequence_lengths,
                                      const Output<Node>& W,
                                      const Output<Node>& R,
                                      const Output<Node>& B,
                                      const Output<Node>& P,
                                      const std::int64_t hidden_size,
                                      const direction lstm_direction,
                                      LSTMWeightsFormat weights_format = LSTMWeightsFormat::IFCO,
                                      const std::vector<float> activations_alpha = {},
                                      const std::vector<float> activations_beta = {},
                                      const std::vector<std::string> activations = {"sigmoid",
                                                                                    "tanh",
                                                                                    "tanh"},
                                      const float clip_threshold = 0,
                                      const bool input_forget = false)
                    : FusedOp({X,
                               initial_hidden_state,
                               initial_cell_state,
                               sequence_lengths,
                               W,
                               R,
                               B,
                               P})
                    , m_activations_alpha(activations_alpha)
                    , m_activations_beta(activations_beta)
                    , m_activations(activations)
                    , m_clip_threshold(clip_threshold)
                    , m_direction(lstm_direction)
                    , m_hidden_size(hidden_size)
                    , m_input_forget(input_forget)
                    , m_weights_format(weights_format)
                {
                    constructor_validate_and_infer_types();
                }

                explicit LSTMSequence(const Output<Node>& X,
                                      const Output<Node>& initial_hidden_state,
                                      const Output<Node>& initial_cell_state,
                                      const Output<Node>& sequence_lengths,
                                      const Output<Node>& W,
                                      const Output<Node>& R,
                                      const Output<Node>& B,
                                      const std::int64_t hidden_size,
                                      const direction lstm_direction,
                                      LSTMWeightsFormat weights_format = LSTMWeightsFormat::IFCO,
                                      const std::vector<float> activations_alpha = {},
                                      const std::vector<float> activations_beta = {},
                                      const std::vector<std::string> activations = {"sigmoid",
                                                                                    "tanh",
                                                                                    "tanh"},
                                      const float clip_threshold = 0,
                                      const bool input_forget = false)
                    : LSTMSequence(
                          X,
                          initial_hidden_state,
                          initial_cell_state,
                          sequence_lengths,
                          W,
                          R,
                          B,
                          Constant::create(
                              element::f32,
                              Shape{(lstm_direction == direction::BIDIRECTIONAL ? 2UL : 1UL),
                                    3UL * static_cast<size_t>(hidden_size)},
                              std::vector<float>{0.f}),
                          hidden_size,
                          lstm_direction,
                          weights_format,
                          activations_alpha,
                          activations_beta,
                          activations,
                          clip_threshold,
                          input_forget)
                {
                }

                bool visit_attributes(AttributeVisitor& visitor) override;
                virtual NodeVector decompose_op() const override;

                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;

                std::vector<float> get_activations_alpha() const { return m_activations_alpha; }
                std::vector<float> get_activations_beta() const { return m_activations_beta; }
                std::vector<std::string> get_activations() const { return m_activations; }
                float get_clip_threshold() const { return m_clip_threshold; }
                direction get_direction() const { return m_direction; }
                std::int64_t get_hidden_size() const { return m_hidden_size; }
                bool get_input_forget() const { return m_input_forget; }
                LSTMWeightsFormat get_weights_format() const { return m_weights_format; }
            private:
                ///
                /// \brief      Gets the masked value according to sequence lenght in a batch.
                ///
                /// \note       Zeros out values or sets them to default value for inputs with
                ///             sequence lenght shorter than currently procssed time step.
                ///
                /// \param[in]  data           The input value.
                /// \param[in]  time_step      The current time step denoting sequence lenght.
                /// \param[in]  batch_axis     The batch axis index of data tensor.
                /// \param[in]  default_value  The default value for masked elements.
                ///
                /// \return     The masked value.
                ///
                std::shared_ptr<Node>
                    get_masked_node(const Output<Node>& data,
                                    std::int32_t time_step,
                                    std::size_t batch_axis = 0,
                                    const Output<Node>& default_value = Output<Node>()) const;

                NodeVector lstm_pass(bool is_reverse = false) const;

                // Split(bi-directional) and squeeze input data to remove 'num_direction' dimension.
                std::shared_ptr<Node> prepare_input(Output<Node> node, bool is_reverse) const;

                std::vector<float> m_activations_alpha;
                std::vector<float> m_activations_beta;
                std::vector<std::string> m_activations;
                float m_clip_threshold;
                direction m_direction;
                std::int64_t m_hidden_size;
                bool m_input_forget;
                LSTMWeightsFormat m_weights_format;
            };
        }
        using v0::LSTMSequence;
    } // namespace op

    std::ostream& operator<<(std::ostream& s, const op::v0::LSTMSequence::direction& type);

    template <>
    class NGRAPH_API AttributeAdapter<op::v0::LSTMSequence::direction>
        : public EnumAttributeAdapterBase<op::v0::LSTMSequence::direction>
    {
    public:
        AttributeAdapter(op::v0::LSTMSequence::direction& value)
            : EnumAttributeAdapterBase<op::v0::LSTMSequence::direction>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{
            "AttributeAdapter<op::v0::LSTMSequence::direction>", 1};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };
} // namespace ngraph
