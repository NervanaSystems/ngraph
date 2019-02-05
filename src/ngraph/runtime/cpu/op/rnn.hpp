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

#pragma once

#include "ngraph/op/op.hpp"
#include "ngraph/runtime/cpu/cpu_backend_visibility.h"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace op
    {
        // This is RNN op, which is formed by the fusion of multiple RNN cells ( LSTM/ GRU/ vanilla RNN)
        // across multiple time slices

        // INPUTS:
        // [0] - {X0, X1...., Xt} input tensor of layout TNC, Shape{sequence length*batch_size, feature_size}
        // [1] - recurrent state tensors {ht_1 | ct_1} of Shape{sequence length*batch_size, feature_size}
        // [2] - initializer for the input weights matrix, used for the linear transformation of the inputs.
        // [3] - initializer for the recurrent weights matrix, used for the linear transformation of the recurrent state.
        // [4] - Initializer for the bias vector w.r.to inputs + hidden state (ibh_bias + hbh_bias)
        // number_of_timesteps - number of unrolled cells up to timestep t.
        // num_gates_per_cell - number of gates per RNN cell, LSTM = 4, GRU = 3, vanilla RNN = 1
        // src_sequence_length - this will be same as number_of_timesteps
        // src_layer_feature_size - feature size w.r.to input tensor
        // src_iter_feature_size - feature size w.r.to hidden state
        // num_cell_states - number of recurrent state tensor states , LSTM = 2, GRU = 1, vanilla RNN = 1

        // OUTPUT VALUE: A tuple with the following structure:
        //   [0] - ht, output tensor with shape (sequence_length*batch_size, feature_size) .
        //   [1] - {ht | ct} output recurrent state tensor with the same shape as states i.e (sequence_length*batch_size, feature_size)

        class Rnn : public Op
        {
        public:
            CPU_BACKEND_API Rnn(std::shared_ptr<Node> src_layer,
                                std::shared_ptr<Node> src_iter,
                                std::shared_ptr<Node> weights_layer,
                                std::shared_ptr<Node> weights_iter,
                                std::shared_ptr<Node> bias,
                                size_t num_timesteps,
                                size_t num_gates_per_cell,
                                size_t src_sequence_length,
                                size_t num_cell_states,
                                size_t direction,
                                size_t num_fused_layers);
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;
            size_t get_num_timesteps() const { return m_num_timesteps; }
            size_t get_src_sequence_length() const { return m_src_sequence_length; }
            size_t get_gates_per_cell() const { return m_num_gates_per_cell; }
            size_t get_batch_size() const { return m_batch_size; }
            size_t get_src_layer_feature_size() const { return m_src_layer_feature_size; }
            size_t get_src_iter_feature_size() const { return m_src_iter_feature_size; }
            size_t get_dst_layer_feature_size() const { return m_dst_layer_feature_size; }
            size_t get_dst_iter_feature_size() const { return m_dst_iter_feature_size; }
            size_t get_num_cell_states() const { return m_num_cell_states; }
            size_t get_direction() const { return m_direction; }
            size_t get_num_fused_layers() const { return m_num_fused_layers; }
        private:
            size_t m_num_timesteps;
            size_t m_num_gates_per_cell;
            size_t m_src_sequence_length;
            size_t m_batch_size;
            size_t m_src_layer_feature_size;
            size_t m_src_iter_feature_size;
            size_t m_dst_layer_feature_size;
            size_t m_dst_iter_feature_size;
            size_t m_num_cell_states;
            size_t m_direction;
            size_t m_num_fused_layers;
        };
    }
}
