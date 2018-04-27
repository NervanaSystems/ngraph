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

#include "ngraph/runtime/cpu/op/lstm.hpp"
#include "ngraph/log.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

shared_ptr<Node> op::Lstm::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 7)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<Lstm>(new_args.at(0),
                             new_args.at(1),
                             new_args.at(2),
                             new_args.at(3),
                             new_args.at(4),
                             new_args.at(5),
                             new_args.at(6),
                             m_lstm_cell_shape);
}

op::Lstm::Lstm(std::shared_ptr<Node> input_xt_1,
               std::shared_ptr<Node> i2h_weights,
               std::shared_ptr<Node> hidden_state_ht_1,
               std::shared_ptr<Node> h2h_weights,
               std::shared_ptr<Node> i2h_bias,
               std::shared_ptr<Node> h2h_bias,
               std::shared_ptr<Node> cell_state_ct_1,
               Shape lstm_cell_shape)
    : RequiresTensorViewArgs("Lstm",
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
    add_output(hidden_state_ht_1->get_element_type(), hidden_state_ht_1->get_shape());
    add_output(cell_state_ct_1->get_element_type(), cell_state_ct_1->get_shape());
}
