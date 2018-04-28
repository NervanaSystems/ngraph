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
                            m_src_layer_feature_size,
                            m_src_iter_feature_size,
                            m_num_rnn_cell_states);
}

op::Rnn::Rnn(std::shared_ptr<Node> src_layer,
             std::shared_ptr<Node> src_iter,
             std::shared_ptr<Node> weights_layer,
             std::shared_ptr<Node> weights_iter,
             std::shared_ptr<Node> bias,
             const int number_of_cells,
             const int number_of_gates_per_cell,
             const int src_seq_length,
             const int src_layer_feature_size,
             const int src_iter_feature_size,
             const int num_rnn_cell_states)
    : RequiresTensorViewArgs("Rnn", {src_layer, src_iter, weights_layer, weights_iter, bias})
    , m_number_of_lstm_cells(number_of_cells)
    , m_number_of_gates_per_cell(number_of_gates_per_cell)
    , m_src_seq_length(src_seq_length)
    , m_src_layer_feature_size(src_layer_feature_size)
    , m_src_iter_feature_size(src_iter_feature_size)
    , m_num_rnn_cell_states(num_rnn_cell_states)
{
    if (src_layer->get_shape().size() != weights_layer->get_shape().size())
    {
        throw ngraph_error("src_layer and i2h weights size dont match");
    }

    if (src_iter->get_shape().size() != weights_iter->get_shape().size())
    {
        throw ngraph_error("src_iter and h2h weights size dont match");
    }

    if (src_layer->get_shape().size() == 2)
    {
        m_batch_size = static_cast<int>(src_layer->get_shape()[0] / number_of_cells);
    }
    else
    {
        throw ngraph_error("src_layer doesnt have a rank 2");
    }

    if (shape_size(src_layer->get_shape()) !=
        m_src_seq_length * m_batch_size * m_src_layer_feature_size)
    {
        std::cout << "shape_size: " << shape_size(src_layer->get_shape()) << std::endl;
        throw ngraph_error("src_layer size is not equal t*n*c");
    }

    if (bias->get_shape()[0] != weights_layer->get_shape()[0] ||
        bias->get_shape()[0] != weights_iter->get_shape()[0])
    {
        throw ngraph_error("bias and weights_shape are not compatible");
    }

    auto et = src_layer->get_element_type();
    for (auto& rnn_input : get_arguments())
    {
        if (rnn_input->get_element_type() != et)
        {
            throw ngraph_error("all rnn inputs must have the same element type");
        }
    }

    add_output(src_layer->get_element_type(),
               Shape{static_cast<unsigned long>(m_number_of_lstm_cells * m_batch_size),
                     static_cast<unsigned long>(m_src_iter_feature_size)});
    add_output(src_layer->get_element_type(),
               Shape{static_cast<unsigned long>(m_num_rnn_cell_states * m_batch_size),
                     static_cast<unsigned long>(m_src_iter_feature_size)});
}
