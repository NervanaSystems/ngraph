/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include "ngraph/op/util/requires_tensor_view_args.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace op
    {
        // In this version of LSTM op:
        //
        // INPUTS:
        // [0] - xt, input tensor of layout TNC, Shape{sequence length*batch_size, feature_size}
        // [1] - initializer for the input weights matrix, used for the linear transformation of the inputs.
        // [2] - ht_1, hidden state of shape (batch_size, feature_size)
        // [3] - initializer for the recurrent weights matrix, used for the linear transformation of the recurrent state.
        // [4] - Initializer for the bias vector w.r.to inputs.
        // [5] - Initializer for the bias vector w.r.to hidden state
        // [6] - ct_1, cell state of shape (batch_size, feature_size)

        // OUTPUT VALUE: A tuple with the following structure:
        //   [0] - ht, output tensor with shape (sequence_length*batch_size, num_hidden) .
        //   [1] - {ht | ct} output recurrent state tensor with the same shape as states

        class Lstm : public util::RequiresTensorViewArgs
        {
        public:
            Lstm(std::shared_ptr<Node> input_xt_1,
                 std::shared_ptr<Node> i2h_weights,
                 std::shared_ptr<Node> hidden_state_ht_1,
                 std::shared_ptr<Node> h2h_weights,
                 std::shared_ptr<Node> i2h_bias,
                 std::shared_ptr<Node> h2h_bias,
                 std::shared_ptr<Node> cell_state_ct_1,
                 Shape lstm_cell_shape);
            Shape get_input_shape() const { return m_shape_input; }
            Shape get_cell_shape() const { return m_lstm_cell_shape; }
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        private:
            Shape m_shape_input;
            Shape m_lstm_cell_shape;
        };
    }
}
