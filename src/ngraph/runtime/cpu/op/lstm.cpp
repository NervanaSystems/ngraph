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

#include "ngraph/runtime/cpu/op/lstm.hpp"
#include "ngraph/log.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Lstm::type_info;

#if MKLDNN_VERSION_MAJOR >= 1
shared_ptr<Node> op::Lstm::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 6)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<Lstm>(new_args.at(0),
                             new_args.at(1),
                             new_args.at(2),
                             new_args.at(3),
                             new_args.at(4),
                             new_args.at(5),
                             m_rnntype);
}

op::Lstm::Lstm(const Output<Node>& src_layer,
               const Output<Node>& src_iter,
               const Output<Node>& src_iter_c,
               const Output<Node>& weights_layer,
               const Output<Node>& weights_iter,
               const Output<Node>& bias,
               ngraph::runtime::cpu::rnn_utils::rnntype rnn_type)
    : Op({src_layer, src_iter, src_iter_c, weights_layer, weights_iter, bias})
    , m_output_tensor_shape(src_layer.get_shape())
    , m_output_cell_shape(src_iter.get_shape())
    , m_num_timesteps(1)
    , m_num_gates_per_cell(4)
    , m_src_sequence_length(1)
    , m_src_layer_feature_size(src_layer.get_shape()[1])
    , m_src_iter_feature_size(src_iter.get_shape()[1])
    , m_num_cell_states(2)
    , m_direction(1)
    , m_num_fused_layers(1)
    , m_rnntype(rnn_type)
{
    constructor_validate_and_infer_types();

    if (src_layer.get_shape().size() != weights_layer.get_shape().size())
    {
        throw ngraph_error("src_layer and i2h weights size dont match");
    }

    if (src_iter.get_shape().size() != weights_iter.get_shape().size())
    {
        throw ngraph_error("src_iter and h2h weights size dont match");
    }

    if (src_layer.get_shape().size() == 2)
    {
        m_batch_size = src_layer.get_shape()[0] / m_num_timesteps;
    }
    else
    {
        throw ngraph_error("src_layer doesnt have a rank 2");
    }

    if (shape_size(src_layer.get_shape()) !=
        m_src_sequence_length * m_batch_size * m_src_layer_feature_size)
    {
        throw ngraph_error("src_layer size is not equal t*n*c");
    }

    if (bias.get_shape()[0] != weights_layer.get_shape()[1] ||
        bias.get_shape()[0] != weights_iter.get_shape()[1])
    {
        throw ngraph_error("bias and weights_shape are not compatible");
    }

    auto et = src_layer.get_element_type();
    for (auto rnn_input : inputs())
    {
        if (rnn_input.get_element_type() != et)
        {
            throw ngraph_error("all rnn inputs must have the same element type");
        }
    }

    set_output_size(3);
    set_output_type(0,
                    src_layer.get_element_type(),
                    Shape{(m_num_timesteps * m_batch_size), m_src_iter_feature_size});
    set_output_type(1, src_layer.get_element_type(), Shape{m_batch_size, m_src_iter_feature_size});
    set_output_type(2, src_layer.get_element_type(), Shape{m_batch_size, m_src_iter_feature_size});
}
#else

shared_ptr<Node> op::Lstm::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 5)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<Lstm>(
        new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), new_args.at(4), m_rnntype);
}

op::Lstm::Lstm(const Output<Node>& src_layer,
               const Output<Node>& src_iter,
               const Output<Node>& weights_layer,
               const Output<Node>& weights_iter,
               const Output<Node>& bias,
               ngraph::runtime::cpu::rnn_utils::rnntype rnn_type)
    : Op({src_layer, src_iter, weights_layer, weights_iter, bias})
    , m_output_tensor_shape(src_layer.get_shape())
    , m_output_cell_shape(src_iter.get_shape())
    , m_num_timesteps(1)
    , m_num_gates_per_cell(4)
    , m_src_sequence_length(1)
    , m_src_layer_feature_size(src_layer.get_shape()[1])
    , m_src_iter_feature_size(src_iter.get_shape()[1])
    , m_num_cell_states(2)
    , m_direction(1)
    , m_num_fused_layers(1)
    , m_rnntype(rnn_type)
{
    constructor_validate_and_infer_types();

    if (src_layer.get_shape().size() != weights_layer.get_shape().size())
    {
        throw ngraph_error("src_layer and i2h weights size dont match");
    }

    if (src_iter.get_shape().size() != weights_iter.get_shape().size())
    {
        throw ngraph_error("src_iter and h2h weights size dont match");
    }

    if (src_layer.get_shape().size() == 2)
    {
        m_batch_size = src_layer.get_shape()[0] / m_num_timesteps;
    }
    else
    {
        throw ngraph_error("src_layer doesnt have a rank 2");
    }

    if (shape_size(src_layer.get_shape()) !=
        m_src_sequence_length * m_batch_size * m_src_layer_feature_size)
    {
        throw ngraph_error("src_layer size is not equal t*n*c");
    }

    if (bias.get_shape()[0] != weights_layer.get_shape()[1] ||
        bias.get_shape()[0] != weights_iter.get_shape()[1])
    {
        throw ngraph_error("bias and weights_shape are not compatible");
    }

    auto et = src_layer.get_element_type();
    for (auto rnn_input : inputs())
    {
        if (rnn_input.get_element_type() != et)
        {
            throw ngraph_error("all rnn inputs must have the same element type");
        }
    }

    set_output_size(2);
    set_output_type(0,
                    src_layer.get_element_type(),
                    Shape{(m_num_timesteps * m_batch_size), m_src_iter_feature_size});
    set_output_type(1,
                    src_layer.get_element_type(),
                    Shape{(m_num_cell_states * m_batch_size), m_src_iter_feature_size});
}
#endif
