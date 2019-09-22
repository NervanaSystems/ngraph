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

#include <mkldnn.hpp>
#include "ngraph/op/op.hpp"
#include "ngraph/runtime/cpu/cpu_backend_visibility.h"
#include "ngraph/runtime/cpu/op/rnn_utils.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace op
    {
        class Lstm : public Op
        {
        public:
            CPU_BACKEND_API
            static constexpr NodeTypeInfo type_info{"Lstm", 0};
            const NodeTypeInfo& get_type_info() const override { return type_info; }
// INPUTS:
// [0] - {Xt} input tensor of layout TNC, Shape{sequence length*batch_size,
//       feature_size}
// [1] - recurrent state tensors {ht_1 | ct_1} of Shape{sequence length*batch_size,
//       feature_size}
// [2] - initializer for the input weights matrix, used for the linear transformation of
//       the inputs.
// [3] - initializer for the recurrent weights matrix, used for the linear
//       transformation of the recurrent state.
// [4] - Initializer for the bias vector w.r.to inputs + hidden state (ibh_bias +
//       hbh_bias)

// OUTPUT VALUE: A tuple with the following structure:
//   [0] - ht, output tensor with shape (sequence_length*batch_size, num_hidden) .
//   [1] - {ht | ct} output recurrent state tensor with the same shape as states

#if MKLDNN_VERSION_MAJOR < 1
            // This version of the LSTM op supports MKLDNN emitter code, this can be used standalone
            // for computing RNN
            // without fusing RNN cell (LSTM)'s across time steps.
            Lstm(const Output<Node>& src_layer,
                 const Output<Node>& src_iter,
                 const Output<Node>& weights_layer,
                 const Output<Node>& weights_iter,
                 const Output<Node>& bias,
                 ngraph::runtime::cpu::rnn_utils::rnntype rnn_type);
#else
            Lstm(const Output<Node>& src_layer,
                 const Output<Node>& src_iter,
                 const Output<Node>& src_iter_c,
                 const Output<Node>& weights_layer,
                 const Output<Node>& weights_iter,
                 const Output<Node>& bias,
                 ngraph::runtime::cpu::rnn_utils::rnntype rnn_type);
#endif
            Shape get_output_tensor_shape() const { return m_output_tensor_shape; }
            Shape get_output_cell_shape() const { return m_output_cell_shape; }
            ngraph::runtime::cpu::rnn_utils::rnntype get_rnn_type() const { return m_rnntype; }
            size_t get_num_timesteps() const { return m_num_timesteps; }
            size_t get_src_sequence_length() const { return m_src_sequence_length; }
            size_t get_gates_per_cell() const { return m_num_gates_per_cell; }
            size_t get_batch_size() const { return m_batch_size; }
            size_t get_src_layer_feature_size() const { return m_src_layer_feature_size; }
            size_t get_src_iter_feature_size() const { return m_src_iter_feature_size; }
            size_t get_num_cell_states() const { return m_num_cell_states; }
            size_t get_direction() const { return m_direction; }
            size_t get_num_fused_layers() const { return m_num_fused_layers; }
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        private:
            Shape m_output_tensor_shape;
            Shape m_output_cell_shape;
            size_t m_num_timesteps;
            size_t m_num_gates_per_cell;
            size_t m_src_sequence_length;
            size_t m_batch_size;
            size_t m_src_layer_feature_size;
            size_t m_src_iter_feature_size;
            size_t m_num_cell_states;
            size_t m_direction;
            size_t m_num_fused_layers;
            ngraph::runtime::cpu::rnn_utils::rnntype m_rnntype;
        };
    }
}
