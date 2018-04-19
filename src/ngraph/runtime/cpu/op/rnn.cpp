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

#include "ngraph/runtime/cpu/op/rnn.hpp"
#include "ngraph/log.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

shared_ptr<Node> op::LSTM::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 7)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<LSTM>(new_args.at(0),
                             new_args.at(1),
                             new_args.at(2),
                             new_args.at(3),
                             new_args.at(4),
                             new_args.at(5),
                             new_args.at(6),
                             m_lstm_cell_shape);
}

op::LSTM::LSTM(std::shared_ptr<Node> input_xt_1,
               std::shared_ptr<Node> i2h_weights,
               std::shared_ptr<Node> hidden_state_ht_1,
               std::shared_ptr<Node> h2h_weights,
               std::shared_ptr<Node> i2h_bias,
               std::shared_ptr<Node> h2h_bias,
               std::shared_ptr<Node> cell_state_ct_1,
               Shape lstm_cell_shape)
    : RequiresTensorViewArgs("LSTM",
                             {input_xt_1,
                              i2h_weights,
                              hidden_state_ht_1,
                              h2h_weights,
                              i2h_bias,
                              h2h_bias,
                              cell_state_ct_1})
    , m_shape_input(hidden_state_ht_1->get_shape())
    , m_lstm_cell_shape(lstm_cell_shape)
{
    add_output(hidden_state_ht_1->get_element_type(), m_shape_input);
    add_output(cell_state_ct_1->get_element_type(), m_lstm_cell_shape);
}

shared_ptr<Node> op::RNN::copy_with_new_args(const NodeVector& new_args) const
{
    return make_shared<RNN>(new_args[0],
                            new_args[1],
                            new_args[2],
                            new_args[3],
                            new_args[4],
                            m_number_of_lstm_cells,
                            m_lstm_output_shape);
}

op::RNN::RNN(std::shared_ptr<Node> src_layer,
             std::shared_ptr<Node> src_iter,
             std::shared_ptr<Node> weights_layer,
             std::shared_ptr<Node> weights_iter,
             std::shared_ptr<Node> bias,
             const int number_of_lstm_cells,
             Shape lstm_output_shape)
    : RequiresTensorViewArgs("RNN", {src_layer, src_iter, weights_layer, weights_iter, bias})
    , m_number_of_lstm_cells(number_of_lstm_cells)
    , m_lstm_output_shape(lstm_output_shape)
{
    for (size_t i = 0; i < number_of_lstm_cells; i++)
    {
        add_output(src_layer->get_element_type(), src_layer->get_shape());
    }
}
