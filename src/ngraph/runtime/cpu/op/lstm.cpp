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

#include "ngraph/runtime/cpu/op/lstm.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

shared_ptr<Node> op::Lstm::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 5)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<Lstm>(
        new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), new_args.at(4));
}

op::Lstm::Lstm(std::shared_ptr<Node> src_layer,
               std::shared_ptr<Node> src_iter,
               std::shared_ptr<Node> weights_layer,
               std::shared_ptr<Node> weights_iter,
               std::shared_ptr<Node> bias)
    : Op("Lstm", check_single_output_args({src_layer, src_iter, weights_layer, weights_iter, bias}))
    , m_output_tensor_shape(src_layer->get_shape())
    , m_output_cell_shape(src_iter->get_shape())
    , m_num_timesteps(1)
    , m_num_gates_per_cell(4)
    , m_src_sequence_length(1)
    , m_src_layer_feature_size(src_layer->get_shape()[1])
    , m_src_iter_feature_size(src_iter->get_shape()[1])
    , m_num_cell_states(2)
    , m_direction(1)
    , m_num_fused_layers(1)
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

    if (shape_size(src_layer->get_shape()) !=
        m_src_sequence_length * m_batch_size * m_src_layer_feature_size)
    {
        throw ngraph_error("src_layer size is not equal t*n*c");
    }

    if (bias->get_shape()[0] != weights_layer->get_shape()[1] ||
        bias->get_shape()[0] != weights_iter->get_shape()[1])
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
                    Shape{(m_num_timesteps * m_batch_size), m_src_iter_feature_size});
    set_output_type(1,
                    src_layer->get_element_type(),
                    Shape{(m_num_cell_states * m_batch_size), m_src_iter_feature_size});
}

void op::Lstm::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
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

    auto lstm_bprop =
        std::make_shared<op::LstmBackprop>(static_pointer_cast<op::Lstm>(shared_from_this()),
                                           src_layer,
                                           src_iter,
                                           weights_layer,
                                           weights_iter,
                                           bias,
                                           fprop_dst_layer,
                                           fprop_dst_iter,
                                           diff_dst_layer,
                                           diff_dst_iter);

    auto diff_src_layer = std::make_shared<op::GetOutputElement>(lstm_bprop, 0);
    auto diff_src_iter = std::make_shared<op::GetOutputElement>(lstm_bprop, 1);
    auto diff_weights_layer = std::make_shared<op::GetOutputElement>(lstm_bprop, 2);
    auto diff_weights_iter = std::make_shared<op::GetOutputElement>(lstm_bprop, 3);
    auto diff_bias = std::make_shared<op::GetOutputElement>(lstm_bprop, 4);

    adjoints.add_delta(src_layer, diff_src_layer);
    adjoints.add_delta(src_iter, diff_src_iter);
    adjoints.add_delta(weights_layer, diff_weights_layer);
    adjoints.add_delta(weights_iter, diff_weights_iter);
    adjoints.add_delta(bias, diff_bias);
}

op::LstmBackprop::LstmBackprop(std::shared_ptr<Node> result_forward,
                               std::shared_ptr<Node> fprop_src_layer,
                               std::shared_ptr<Node> fprop_src_iter,
                               std::shared_ptr<Node> fprop_weights_layer,
                               std::shared_ptr<Node> fprop_weights_iter,
                               std::shared_ptr<Node> fprop_bias,
                               std::shared_ptr<Node> fprop_dst_layer,
                               std::shared_ptr<Node> fprop_dst_iter,
                               std::shared_ptr<Node> diff_dst_layer,
                               std::shared_ptr<Node> diff_dst_iter)
    : Op("LstmBackprop",
         check_single_output_args({fprop_src_layer,
                                   fprop_src_iter,
                                   fprop_weights_layer,
                                   fprop_weights_iter,
                                   fprop_bias,
                                   fprop_dst_layer,
                                   fprop_dst_iter,
                                   diff_dst_layer,
                                   diff_dst_iter}))
    , m_fprop_node(result_forward)
{
    auto rnn_node = static_cast<const ngraph::op::Lstm*>(result_forward.get());
    m_rnn_attributes.timestep = rnn_node->get_src_sequence_length();
    m_rnn_attributes.batch = rnn_node->get_batch_size();
    m_rnn_attributes.states = rnn_node->get_num_cell_states();
    m_rnn_attributes.layer = rnn_node->get_num_fused_layers();
    m_rnn_attributes.direction = rnn_node->get_direction();
    m_rnn_attributes.gates = rnn_node->get_gates_per_cell();
    m_rnn_attributes.slc = rnn_node->get_src_layer_feature_size();
    m_rnn_attributes.sic = rnn_node->get_src_iter_feature_size();

    set_output_size(5);
    set_output_type(0, fprop_src_layer->get_element_type(), fprop_src_layer->get_shape());
    set_output_type(1, fprop_src_iter->get_element_type(), fprop_src_iter->get_shape());

    // mkldnn rnn bprop outputs in the order ldgoi
    auto wei_layer_shape = fprop_weights_layer->get_shape();
    set_output_type(
        2, fprop_weights_layer->get_element_type(), Shape{wei_layer_shape[1], wei_layer_shape[0]});

    auto wei_iter_shape = fprop_weights_iter->get_shape();
    set_output_type(
        3, fprop_weights_iter->get_element_type(), Shape{wei_iter_shape[1], wei_iter_shape[0]});

    set_output_type(4, fprop_bias->get_element_type(), fprop_bias->get_shape());
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::LstmBackprop::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 9)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    auto lstm_fprop_node = this->get_fprop_node();
    return make_shared<LstmBackprop>(lstm_fprop_node,
                                     new_args[0],
                                     new_args[1],
                                     new_args[2],
                                     new_args[3],
                                     new_args[4],
                                     new_args[5],
                                     new_args[6],
                                     new_args[7],
                                     new_args[8]);
}
