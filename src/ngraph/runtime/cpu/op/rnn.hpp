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
        class LSTM : public util::RequiresTensorViewArgs
        {
        public:
            LSTM(std::shared_ptr<Node> input_xt_1,
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
            // virtual void generate_adjoints(autodiff::Adjoints& adjoints,
            //                                const std::shared_ptr<Node>& delta) override;

        private:
            Shape m_shape_input;
            Shape m_lstm_cell_shape;
        };

        class RNN : public util::RequiresTensorViewArgs
        {
        public:
            RNN(std::shared_ptr<Node> src_layer,
                std::shared_ptr<Node> src_iter,
                std::shared_ptr<Node> weights_layer,
                std::shared_ptr<Node> weights_iter,
                std::shared_ptr<Node> bias,
                const int number_of_lstm_cells,
                Shape lstm_output_shape);
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;
            int get_num_of_lstm_cells() { return m_number_of_lstm_cells; }
        private:
            int m_number_of_lstm_cells;
            Shape m_lstm_output_shape;
        };
    }
}
