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
        class Rnn : public util::RequiresTensorViewArgs
        {
        public:
            Rnn(std::shared_ptr<Node> src_layer,
                std::shared_ptr<Node> src_iter,
                std::shared_ptr<Node> weights_layer,
                std::shared_ptr<Node> weights_iter,
                std::shared_ptr<Node> bias,
                const int number_of_cells,
                const int number_of_gates_per_cell,
                const int src_seq_length,
                const int input_feature_size,
                const int num_rnn_cell_states);
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;
            int get_num_of_lstm_cells_fused() const { return m_number_of_lstm_cells; }
            int get_src_sequence_length() const { return m_src_seq_length; }
            int get_gates_per_cell() const { return m_number_of_gates_per_cell; }
            int get_batch_size() const { return m_batch_size; }
            int get_input_feature_size() const { return m_input_feature_size; }
            int get_num_rnn_cell_states() const { return m_num_rnn_cell_states; }
        private:
            int m_number_of_lstm_cells;
            int m_number_of_gates_per_cell;
            int m_src_seq_length;
            int m_batch_size;
            int m_input_feature_size;
            int m_num_rnn_cell_states;
        };
    }
}
