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

#pragma once

#include "ngraph/op/experimental/quantized_conv.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Convolution + bias forward prop for batched convolution operation.
        class QuantizedConvolutionBias : public Op
        {
        public:
            QuantizedConvolutionBias(const std::shared_ptr<op::QuantizedConvolution>& qconv,
                                     const std::shared_ptr<Node>& bias,
                                     const bool with_relu = false);

            QuantizedConvolutionBias(const std::shared_ptr<Node>& data_batch,
                                     const std::shared_ptr<Node>& filters,
                                     const std::shared_ptr<Node>& bias,
                                     const Strides& window_movement_strides,
                                     const Strides& window_dilation_strides,
                                     const CoordinateDiff& padding_below,
                                     const CoordinateDiff& padding_above,
                                     const Strides& data_dilation_strides,
                                     const std::shared_ptr<Node> scale,
                                     const std::shared_ptr<Node> bias_scale,
                                     const bool with_relu = false);

            const Strides& get_window_movement_strides() const { return m_window_movement_strides; }
            const Strides& get_window_dilation_strides() const { return m_window_dilation_strides; }
            const CoordinateDiff& get_padding_below() const { return m_padding_below; }
            const CoordinateDiff& get_padding_above() const { return m_padding_above; }
            const Strides& get_data_dilation_strides() const { return m_data_dilation_strides; }
            float get_scale() const { return m_scale; }
            float get_bias_scale() const { return m_bias_scale; }
            std::shared_ptr<Node> get_bias() { return get_argument(2); }
            std::shared_ptr<Node> get_filters() { return get_argument(1); }
            std::shared_ptr<Node> get_data_batch() { return get_argument(0); }
            bool with_relu() const { return m_with_relu; }
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        protected:
            Strides m_window_movement_strides;
            Strides m_window_dilation_strides;
            CoordinateDiff m_padding_below;
            CoordinateDiff m_padding_above;
            Strides m_data_dilation_strides;
            bool m_with_relu;
            float m_scale;
            float m_bias_scale;
        };
    }
}
