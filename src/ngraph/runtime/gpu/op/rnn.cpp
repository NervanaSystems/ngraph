//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "ngraph/runtime/gpu/op/rnn.hpp"
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::gpu::Rnn::type_info;

shared_ptr<Node> op::gpu::Rnn::copy_with_new_args(const NodeVector& new_args) const
{
    NGRAPH_CHECK(new_args.size() == 4, "Incorrect number of new arguments");

    return make_shared<Rnn>(new_args[0],
                            new_args[1],
                            new_args[2],
                            new_args[3],
                            m_num_timesteps,
                            m_num_gates_per_cell,
                            m_src_sequence_length,
                            m_src_layer_feature_size,
                            m_src_iter_feature_size,
                            m_direction,
                            m_num_fused_layers);
}

op::gpu::Rnn::Rnn(std::shared_ptr<Node> src_layer,
                  std::shared_ptr<Node> src_iter,
                  std::shared_ptr<Node> params,
                  std::shared_ptr<Node> state_iter,
                  const int num_timesteps,
                  const int num_gates_per_cell,
                  const int src_sequence_length,
                  const int src_layer_feature_size,
                  const int src_iter_feature_size,
                  const int direction,
                  const int num_fused_layers)
    : Op("Rnn", {src_layer, src_iter, params, state_iter})
    , m_num_timesteps(num_timesteps)
    , m_num_gates_per_cell(num_gates_per_cell)
    , m_src_sequence_length(src_sequence_length)
    , m_src_layer_feature_size(src_layer_feature_size)
    , m_src_iter_feature_size(src_iter_feature_size)
    , m_direction(direction)
    , m_num_fused_layers(num_fused_layers)
{
    NGRAPH_CHECK(src_layer->get_shape().size() == 2, "src_layer doesnt have a rank 2");

    m_batch_size = static_cast<int>(src_layer->get_shape()[0] / num_timesteps);

    NGRAPH_CHECK(shape_size(src_layer->get_shape()) ==
                     m_src_sequence_length * m_batch_size * m_src_layer_feature_size,
                 "src_layer size is not equal t*n*c");

    auto et = src_layer->get_element_type();
    for (auto& rnn_input : get_arguments())
    {
        if (rnn_input->get_element_type() != et)
        {
            throw ngraph_error("all rnn inputs must have the same element type");
        }
    }

    set_output_size(3);
    set_output_type(0,
                    src_layer->get_element_type(),
                    Shape{static_cast<unsigned long>(m_direction * m_num_timesteps * m_batch_size),
                          static_cast<unsigned long>(m_src_iter_feature_size)});
    set_output_type(
        1,
        src_layer->get_element_type(),
        Shape{static_cast<unsigned long>(m_direction * m_num_fused_layers * m_batch_size),
              static_cast<unsigned long>(m_src_iter_feature_size)});
    set_output_type(
        2,
        src_layer->get_element_type(),
        Shape{static_cast<unsigned long>(m_direction * m_num_fused_layers * m_batch_size),
              static_cast<unsigned long>(m_src_iter_feature_size)});
}
