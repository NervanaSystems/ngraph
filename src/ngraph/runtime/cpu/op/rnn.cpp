//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include "ngraph/runtime/cpu/op/rnn.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/get_output_element.hpp"
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
                            m_num_timesteps,
                            m_num_gates_per_cell,
                            m_src_sequence_length,
                            m_num_cell_states,
                            m_direction,
                            m_num_fused_layers);
}

op::Rnn::Rnn(std::shared_ptr<Node> src_layer,
             std::shared_ptr<Node> src_iter,
             std::shared_ptr<Node> weights_layer,
             std::shared_ptr<Node> weights_iter,
             std::shared_ptr<Node> bias,
             size_t num_timesteps,
             size_t num_gates_per_cell,
             size_t src_sequence_length,
             size_t num_cell_states,
             size_t direction,
             size_t num_fused_layers)
    : Op("Rnn", check_single_output_args({src_layer, src_iter, weights_layer, weights_iter, bias}))
    , m_num_timesteps(num_timesteps)
    , m_num_gates_per_cell(num_gates_per_cell)
    , m_src_sequence_length(src_sequence_length)
    , m_num_cell_states(num_cell_states)
    , m_direction(direction)
    , m_num_fused_layers(num_fused_layers)
{
    constructor_validate_and_infer_types();
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
        m_batch_size = src_layer->get_shape()[0] / m_num_timesteps;
    }
    else
    {
        throw ngraph_error("src_layer doesnt have a rank 2");
    }

    m_dst_iter_feature_size = weights_iter->get_shape()[1] / (m_num_gates_per_cell);
    m_dst_layer_feature_size = weights_layer->get_shape()[1] / (m_num_gates_per_cell);
    m_src_iter_feature_size = weights_iter->get_shape()[0] / (m_direction * m_num_fused_layers);
    m_src_layer_feature_size = weights_layer->get_shape()[0] / (m_direction * m_num_fused_layers);

    if (shape_size(src_layer->get_shape()) !=
        m_src_sequence_length * m_batch_size * m_src_layer_feature_size)
    {
        throw ngraph_error("src_layer size is not equal t*n*c");
    }

    if ((bias->get_shape()[0] / m_num_fused_layers) != (weights_layer->get_shape()[1]) ||
        (bias->get_shape()[0] / m_num_fused_layers) != (weights_iter->get_shape()[1]))
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

    set_output_size(2);
    set_output_type(0,
                    src_layer->get_element_type(),
                    Shape{(m_direction * m_num_timesteps * m_batch_size), m_src_iter_feature_size});
    set_output_type(1,
                    src_layer->get_element_type(),
                    Shape{(m_num_cell_states * m_direction * m_num_fused_layers * m_batch_size),
                          m_src_iter_feature_size});
}

void op::Rnn::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto diff_dst_layer = deltas.at(0);
    auto diff_dst_iter = deltas.at(1);

    auto src_layer = get_argument(0);
    auto src_iter = get_argument(1);
    auto weights_layer = get_argument(2);
    auto weights_iter = get_argument(3);
    auto bias = get_argument(4);

    auto goes = op::get_output_elements(shared_from_this());
    auto fprop_dst_layer = goes.at(0);
    auto fprop_dst_iter = goes.at(1);

    auto rnn_bprop = std::make_shared<op::RnnBackprop>(static_pointer_cast<op::Rnn>(shared_from_this()),
                                                       src_layer,
                                                       src_iter,
                                                       weights_layer,
                                                       weights_iter,
                                                       bias,
                                                       fprop_dst_layer,
                                                       fprop_dst_iter,
                                                       diff_dst_layer,
                                                       diff_dst_iter);

    auto diff_src_layer = std::make_shared<op::GetOutputElement>(rnn_bprop, 0);
    auto diff_src_iter = std::make_shared<op::GetOutputElement>(rnn_bprop, 1);
    auto diff_weights_layer = std::make_shared<op::GetOutputElement>(rnn_bprop, 2);
    auto diff_weights_iter = std::make_shared<op::GetOutputElement>(rnn_bprop, 3);
    auto diff_bias = std::make_shared<op::GetOutputElement>(rnn_bprop, 4);

    adjoints.add_delta(src_layer, diff_src_layer);
    adjoints.add_delta(src_iter, diff_src_iter);
    adjoints.add_delta(weights_layer, diff_weights_layer);
    adjoints.add_delta(weights_iter, diff_weights_iter);
    adjoints.add_delta(bias, diff_bias);
}

op::RnnBackprop::RnnBackprop(std::shared_ptr<Node> result_forward,
                             std::shared_ptr<Node> fprop_src_layer,
                             std::shared_ptr<Node> fprop_src_iter,
                             std::shared_ptr<Node> fprop_weights_layer,
                             std::shared_ptr<Node> fprop_weights_iter,
                             std::shared_ptr<Node> fprop_bias,
                             std::shared_ptr<Node> fprop_dst_layer,
                             std::shared_ptr<Node> fprop_dst_iter,
                             std::shared_ptr<Node> diff_dst_layer,
                             std::shared_ptr<Node> diff_dst_iter)
    : Op("RnnBackprop",
         check_single_output_args({result_forward,
                                   fprop_src_layer,
                                   fprop_src_iter,
                                   fprop_weights_layer,
                                   fprop_weights_iter,
                                   fprop_bias,
                                   fprop_dst_layer,
                                   fprop_dst_iter,
                                   diff_dst_layer,
                                   diff_dst_iter}))
{
    set_output_size(5);
    set_output_type(0, fprop_src_layer->get_element_type(), fprop_src_layer->get_shape());
    set_output_type(1, fprop_src_layer->get_element_type(), fprop_src_iter->get_shape());
    set_output_type(2, fprop_src_layer->get_element_type(), fprop_weights_layer->get_shape());
    set_output_type(3, fprop_src_layer->get_element_type(), fprop_weights_iter->get_shape());
    set_output_type(4, fprop_src_layer->get_element_type(), fprop_bias->get_shape());
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::RnnBackprop::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 9)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<RnnBackprop>(new_args[0],
                                    new_args[1],
                                    new_args[2],
                                    new_args[3],
                                    new_args[4],
                                    new_args[5],
                                    new_args[6],
                                    new_args[7],
                                    new_args[8],
                                    new_args[9]);
}
