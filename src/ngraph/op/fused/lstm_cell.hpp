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

#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/util/activation_functions.hpp"
#include "ngraph/op/util/fused_op.hpp"
#include "ngraph/op/util/rnn_cell_base.hpp"

namespace ngraph
{
    namespace op
    {
        ///
        /// \brief      Class for lstm cell node.
        ///
        /// \note       It follows notation and equations defined as in ONNX standard:
        ///             https://github.com/onnx/onnx/blob/master/docs/Operators.md#LSTM
        ///
        ///             Note this class represents only single *cell* and not whole LSTM *layer*.
        ///
        class LSTMCell : public util::FusedOp, public util::RNNCellBase
        {
        public:
            NGRAPH_API
            static constexpr NodeTypeInfo type_info{"LSTMCell", 0};
            const NodeTypeInfo& get_type_info() const override { return type_info; }
            ///
            /// \brief      Constructs LSTMCell node.
            ///
            /// \param[in]  X            The input tensor with shape: [batch_size, input_size].
            /// \param[in]  W            The weight tensor with shape: [4*hidden_size, input_size].
            /// \param[in]  R            The recurrence weight tensor with shape:
            ///                             [4*hidden_size, hidden_size].
            /// \param[in]  H_t          The hidden state tensor at current time step with shape:
            ///                             [batch_size, hidden_size].
            /// \param[in]  C_t          The cell state tensor at current time step with shape:
            ///                             [batch_size, hidden_size].
            /// \param[in]  hidden_size  The number of hidden units for recurrent cell.
            ///
            LSTMCell(const Output<Node>& X,
                     const Output<Node>& W,
                     const Output<Node>& R,
                     const Output<Node>& H_t,
                     const Output<Node>& C_t,
                     std::size_t hidden_size);

            ///
            /// \brief      Constructs LSTMCell node.
            ///
            /// \param[in]  X                 The input tensor with shape: [batch_size, input_size].
            /// \param[in]  W                 The weight tensor with shape: [4*hidden_size,
            ///                                                              input_size].
            /// \param[in]  R                 The recurrence weight tensor with shape:
            ///                               [4*hidden_size, hidden_size].
            /// \param[in]  H_t               The hidden state tensor at current time step with
            ///                               shape: [batch_size, hidden_size].
            /// \param[in]  C_t               The cell state tensor at current time step with shape:
            ///                               [batch_size, hidden_size].
            /// \param[in]  hidden_size       The number of hidden units for recurrent cell.
            /// \param[in]  activations       The vector of activation functions used inside
            ///                               recurrent cell.
            /// \param[in]  activation_alpha  The vector of alpha parameters for activation
            ///                               functions in order respective to activation list.
            /// \param[in]  activation_beta   The vector of beta parameters for activation functions
            ///                               in order respective to activation list.
            /// \param[in]  clip              The value defining clipping range [-clip, clip] on
            ///                               input of activation functions.
            /// \param[in]  input_forget      Controls coupling input and forget gates.
            ///
            LSTMCell(const Output<Node>& X,
                     const Output<Node>& W,
                     const Output<Node>& R,
                     const Output<Node>& H_t,
                     const Output<Node>& C_t,
                     std::size_t hidden_size,
                     const std::vector<std::string>& activations,
                     const std::vector<float>& activation_alpha,
                     const std::vector<float>& activation_beta,
                     float clip,
                     bool input_forget);

            ///
            /// \brief      Constructs LSTMCell node.
            ///
            /// \param[in]  X                 The input tensor with shape: [batch_size, input_size].
            /// \param[in]  W                 The weight tensor with shape: [4*hidden_size,
            ///                                                              input_size].
            /// \param[in]  R                 The recurrence weight tensor with shape:
            ///                               [4*hidden_size, hidden_size].
            /// \param[in]  H_t               The hidden state tensor at current time step with
            ///                               shape: [batch_size, hidden_size].
            /// \param[in]  C_t               The cell state tensor at current time step with
            ///                               shape: [batch_size, hidden_size].
            /// \param[in]  hidden_size       The number of hidden units for recurrent cell.
            /// \param[in]  B                 The bias tensor for input gate with shape:
            ///                               [8*hidden_size].
            /// \param[in]  P                 The weight tensor for peepholes with shape:
            ///                               [3*hidden_size] - 3 equals to only iof gates.
            /// \param[in]  activations       The vector of activation functions used inside
            ///                               recurrent cell.
            /// \param[in]  activation_alpha  The vector of alpha parameters for activation
            ///                               functions in order respective to activation list.
            /// \param[in]  activation_beta   The vector of beta parameters for activation functions
            ///                               in order respective to activation list.
            /// \param[in]  clip              The value defining clipping range [-clip, clip] on
            ///                               input of activation functions.
            /// \param[in]  input_forget      Controls coupling input and forget gates.
            ///
            LSTMCell(const Output<Node>& X,
                     const Output<Node>& W,
                     const Output<Node>& R,
                     const Output<Node>& H_t,
                     const Output<Node>& C_t,
                     std::size_t hidden_size,
                     const Output<Node>& B,
                     const Output<Node>& P,
                     const std::vector<std::string>& activations =
                         std::vector<std::string>{"sigmoid", "tanh", "tanh"},
                     const std::vector<float>& activation_alpha = {},
                     const std::vector<float>& activation_beta = {},
                     float clip = 0.f,
                     bool input_forget = false);

            virtual void pre_validate_and_infer_types() override;
            virtual NodeVector decompose_op() const override;
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            bool get_input_forget() const { return m_input_forget; }
        private:
            Output<Node> get_bias() const;
            NodeVector get_peephole_weights() const;

            /// brief Add and initialize bias input to all zeros.
            void add_default_bias_input();
            /// brief Add and initialize peepholes weights input to all zeros.
            void add_default_peepholes_input();

            ///
            /// \brief The Activation function f.
            ///
            util::ActivationFunction m_activation_f;
            ///
            /// \brief The Activation function g.
            ///
            util::ActivationFunction m_activation_g;
            ///
            /// \brief The Activation function h.
            ///
            util::ActivationFunction m_activation_h;
            ///
            /// \brief      Controls whether to couple input and forget gates.
            ///
            bool m_input_forget = false;

            static constexpr std::size_t s_gates_count{4};
            static constexpr std::size_t s_peepholes_count{3};
        };
    }
}
