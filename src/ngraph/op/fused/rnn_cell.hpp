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

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "ngraph/node.hpp"
#include "ngraph/op/util/activation_functions.hpp"
#include "ngraph/op/util/fused_op.hpp"
#include "ngraph/op/util/rnn_cell_base.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            ///
            /// \brief      Class for single RNN cell node.
            ///
            /// \note       It follows notation and equations defined as in ONNX standard:
            ///             https://github.com/onnx/onnx/blob/master/docs/Operators.md#RNN
            ///
            /// \note       It calculates following equations:
            ///
            ///             Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
            ///
            ///             *       - Is a dot product,
            ///             f       - is activation functions.
            ///
            /// \note       This class represents only single *cell* (for current time step) and not
            /// the
            ///             whole LSTM Sequence layer
            ///
            /// \sa         LSTMSequence, LSTMCell, GRUCell
            ///
            class NGRAPH_API RNNCell : public util::FusedOp, public util::RNNCellBase
            {
            public:
                static constexpr NodeTypeInfo type_info{"RNNCell", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                RNNCell() = default;
                ///
                /// \brief      Constructs RNNCell node.
                ///
                /// \param[in]  X                     The input tensor with shape: [batch_size,
                ///                                   input_size].
                /// \param[in]  initial_hidden_state  The hidden state tensor at current time step
                /// with
                ///                                   shape: [batch_size, hidden_size].
                /// \param[in]  W                     The weight tensor with shape: [hidden_size,
                ///                                   input_size].
                /// \param[in]  R                     The recurrence weight tensor with shape:
                ///                                   [hidden_size, hidden_size].
                /// \param[in]  hidden_size           The number of hidden units for recurrent cell.
                /// \param[in]  activations           The vector of activation functions used inside
                ///                                   recurrent cell.
                /// \param[in]  activations_alpha     The vector of alpha parameters for activation
                ///                                   functions in order respective to activation
                ///                                   list.
                /// \param[in]  activations_beta      The vector of beta parameters for activation
                ///                                   functions in order respective to activation
                ///                                   list.
                /// \param[in]  clip                  The value defining clipping range [-clip,
                /// clip] on
                ///                                   input of activation functions.
                ///
                RNNCell(
                    const Output<Node>& X,
                    const Output<Node>& initial_hidden_state,
                    const Output<Node>& W,
                    const Output<Node>& R,
                    std::size_t hidden_size,
                    const std::vector<std::string>& activations = std::vector<std::string>{"tanh"},
                    const std::vector<float>& activations_alpha = {},
                    const std::vector<float>& activations_beta = {},
                    float clip = 0.f);

                ///
                /// \brief      Constructs RNNCell node.
                ///
                /// \param[in]  X                     The input tensor with shape: [batch_size,
                ///                                   input_size].
                /// \param[in]  initial_hidden_state  The hidden state tensor at current time step
                /// with
                ///                                   shape: [batch_size, hidden_size].
                /// \param[in]  W                     The weight tensor with shape: [hidden_size,
                ///                                   input_size].
                /// \param[in]  R                     The recurrence weight tensor with shape:
                ///                                   [hidden_size, hidden_size].
                /// \param[in]  B                     The bias tensor for input gate with shape:
                ///                                   [hidden_size].
                /// \param[in]  hidden_size           The number of hidden units for recurrent cell.
                /// \param[in]  activations           The vector of activation functions used inside
                ///                                   recurrent cell.
                /// \param[in]  activations_alpha     The vector of alpha parameters for activation
                ///                                   functions in order respective to activation
                ///                                   list.
                /// \param[in]  activations_beta      The vector of beta parameters for activation
                ///                                   functions in order respective to activation
                ///                                   list.
                /// \param[in]  clip                  The value defining clipping range [-clip,
                /// clip] on
                ///                                   input of activation functions.
                ///
                RNNCell(
                    const Output<Node>& X,
                    const Output<Node>& initial_hidden_state,
                    const Output<Node>& W,
                    const Output<Node>& R,
                    const Output<Node>& B,
                    std::size_t hidden_size,
                    const std::vector<std::string>& activations = std::vector<std::string>{"tanh"},
                    const std::vector<float>& activations_alpha = {},
                    const std::vector<float>& activations_beta = {},
                    float clip = 0.f);

                virtual void pre_validate_and_infer_types() override;
                virtual NodeVector decompose_op() const override;
                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;

            private:
                ///
                /// \brief      Creates the default bias input initialized with zeros.
                ///
                /// \return     The object of Output class.
                ///
                Output<Node> get_default_bias_input() const;

                ///
                /// \brief The Activation function f.
                ///
                util::ActivationFunction m_activation_f;

                static constexpr std::size_t s_gates_count{1};
            };
        }
        using v0::RNNCell;
    } // namespace op
} // namespace ngraph
