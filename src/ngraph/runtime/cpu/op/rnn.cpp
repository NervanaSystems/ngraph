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

shared_ptr<Node> op::Rnn::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 5)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<Rnn>(new_args[0],
                            new_args[1],
                            new_args[2],
                            new_args[3],
                            new_args[4],
                            m_number_of_lstm_cells,
                            m_number_of_gates_per_cell,
                            m_src_seq_length,
                            m_input_feature_size);
    //m_output_feature_size);
}

op::Rnn::Rnn(std::shared_ptr<Node> src_layer,
             std::shared_ptr<Node> src_iter,
             std::shared_ptr<Node> weights_layer,
             std::shared_ptr<Node> weights_iter,
             std::shared_ptr<Node> bias,
             const int number_of_cells,
             const int number_of_gates_per_cell,
             const int src_seq_length,
             const int input_feature_size)
    //const int output_feature_size)
    : RequiresTensorViewArgs("Rnn", {src_layer, src_iter, weights_layer, weights_iter, bias}),
      m_number_of_lstm_cells(number_of_cells),
      m_number_of_gates_per_cell(number_of_gates_per_cell),
      m_src_seq_length(src_seq_length),
      m_input_feature_size(input_feature_size)
//, m_output_feature_size(output_feature_size),
{
    if (src_layer->get_shape().size() == 2)
    {
        m_batch_size = static_cast<int>(src_layer->get_shape()[0] / number_of_cells);
    }
    else
    {
        throw ngraph_error("src_layer doesnt have a rank 2");
    }

    add_output(src_layer->get_element_type(), src_layer->get_shape());
    add_output(src_layer->get_element_type(),
               Shape{static_cast<unsigned long>(2 * m_batch_size),
                     static_cast<unsigned long>(input_feature_size)});
}
