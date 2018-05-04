/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#pragma once

#include "ngraph/op/convolution.hpp"
#include "ngraph/op/util/requires_tensor_view_args.hpp"
#include "ngraph/runtime/cpu/op/conv_bias.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Relu(Convolution) forward prop for batched convolution operation.
        class ConvolutionRelu : public util::RequiresTensorViewArgs
        {
        public:
            ConvolutionRelu(const std::shared_ptr<op::Convolution>& conv);

            ConvolutionRelu(const std::shared_ptr<Node>& data_batch,
                            const std::shared_ptr<Node>& filters,
                            const Strides& window_movement_strides,
                            const Strides& window_dilation_strides,
                            const CoordinateDiff& padding_below,
                            const CoordinateDiff& padding_above,
                            const Strides& data_dilation_strides);

            const Strides& get_window_movement_strides() const { return m_window_movement_strides; }
            const Strides& get_window_dilation_strides() const { return m_window_dilation_strides; }
            const CoordinateDiff& get_padding_below() const { return m_padding_below; }
            const CoordinateDiff& get_padding_above() const { return m_padding_above; }
            const Strides& get_data_dilation_strides() const { return m_data_dilation_strides; }
            std::shared_ptr<Node> get_filters() { return get_argument(1); }
            std::shared_ptr<Node> get_data_batch() { return get_argument(0); }
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        protected:
            Strides m_window_movement_strides;
            Strides m_window_dilation_strides;
            CoordinateDiff m_padding_below;
            CoordinateDiff m_padding_above;
            Strides m_data_dilation_strides;
        };

        /// \brief Relu(Convolution) forward prop for batched convolution operation with bias
        class ConvolutionBiasRelu : public util::RequiresTensorViewArgs
        {
        public:
            ConvolutionBiasRelu(const std::shared_ptr<op::ConvolutionBias>& conv);

            ConvolutionBiasRelu(const std::shared_ptr<Node>& data_batch,
                                const std::shared_ptr<Node>& filters,
                                const std::shared_ptr<Node>& bias,
                                const Strides& window_movement_strides,
                                const Strides& window_dilation_strides,
                                const CoordinateDiff& padding_below,
                                const CoordinateDiff& padding_above,
                                const Strides& data_dilation_strides);

            const Strides& get_window_movement_strides() const { return m_window_movement_strides; }
            const Strides& get_window_dilation_strides() const { return m_window_dilation_strides; }
            const CoordinateDiff& get_padding_below() const { return m_padding_below; }
            const CoordinateDiff& get_padding_above() const { return m_padding_above; }
            const Strides& get_data_dilation_strides() const { return m_data_dilation_strides; }
            std::shared_ptr<Node> get_filters() { return get_argument(1); }
            std::shared_ptr<Node> get_data_batch() { return get_argument(0); }
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        protected:
            Strides m_window_movement_strides;
            Strides m_window_dilation_strides;
            CoordinateDiff m_padding_below;
            CoordinateDiff m_padding_above;
            Strides m_data_dilation_strides;
        };
    }
}
