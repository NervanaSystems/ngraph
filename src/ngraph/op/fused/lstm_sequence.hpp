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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ngraph/node.hpp"
#include "ngraph/op/util/fused_op.hpp"

namespace ngraph
{
    namespace op
    {
        ///
        /// \brief      Class for lstm sequence node.
        ///
        /// \note       It follows notation and equations defined as in ONNX standard:
        ///             https://github.com/onnx/onnx/blob/master/docs/Operators.md#LSTM
        ///
        class LSTMSequence : public util::FusedOp
        {
        public:
            NGRAPH_API
            static const std::string type_name;
            const std::string& description() const override { return type_name; }
            LSTMSequence() = default;
            explicit LSTMSequence(const Output<Node>& X,
                                  const Output<Node>& W,
                                  const Output<Node>& R,
                                  const Output<Node>& B,
                                  const Output<Node>& P,
                                  const Output<Node>& initial_h,
                                  const Output<Node>& initial_c,
                                  const Output<Node>& seq_lengths,
                                  const std::vector<float> activations_alpha,
                                  const std::vector<float> activations_beta,
                                  const std::vector<std::string> activations,
                                  const float clip_threshold,
                                  const std::string direction,
                                  const std::int64_t hidden_size,
                                  const bool input_forget)
                : FusedOp({X, W, R, B, P, initial_h, initial_c, seq_lengths})
                , m_activations_alpha(activations_alpha)
                , m_activations_beta(activations_beta)
                , m_activations(activations)
                , m_clip_threshold(clip_threshold)
                , m_direction(direction)
                , m_hidden_size(hidden_size)
                , m_input_forget(input_forget)
            {
            }

            virtual NodeVector decompose_op() const override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        private:
            ///
            /// \brief      Gets the masked node according to sequence lenght in a batch.
            ///
            /// \note       Zeros out values or sets them to default value for inputs with
            ///             sequence lenght shorter than currently procssed time step.
            ///
            /// \param[in]  data           The input node.
            /// \param[in]  time_step      The current time step denoting sequence lenght.
            /// \param[in]  batch_axis     The batch axis index of data tensor.
            /// \param[in]  default_value  The default value for masked elements.
            ///
            /// \return     The masked node.
            ///
            std::shared_ptr<Node> get_masked_node(const std::shared_ptr<Node>& data,
                                                  std::int32_t time_step,
                                                  std::size_t batch_axis = 0,
                                                  const std::shared_ptr<Node>& default_value = {
                                                      nullptr}) const;

            NodeVector lstm_pass(bool is_reverse = false) const;

            std::shared_ptr<Node> prepare_input(Output<Node> node, bool is_reverse) const;

            const std::vector<float> m_activations_alpha;
            const std::vector<float> m_activations_beta;
            const std::vector<std::string> m_activations;
            const float m_clip_threshold;
            const std::string m_direction;
            const std::int64_t m_hidden_size;
            const bool m_input_forget;
        };
    } // namespace op
} // namespace ngraph
