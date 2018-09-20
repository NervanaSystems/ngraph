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

#include "ngraph/op/convolution.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Convolution + bias forward prop for batched convolution operation.
        class ConvolutionBias : public Op
        {
        public:
            ConvolutionBias(const std::shared_ptr<op::Convolution>& conv,
                            const std::shared_ptr<Node>& bias,
                            const bool with_relu = false);

            ConvolutionBias(const std::shared_ptr<Node>& data_batch,
                            const std::shared_ptr<Node>& filters,
                            const std::shared_ptr<Node>& bias,
                            const Strides& window_movement_strides,
                            const Strides& window_dilation_strides,
                            const CoordinateDiff& padding_below,
                            const CoordinateDiff& padding_above,
                            const Strides& data_dilation_strides,
                            const bool with_relu = false);

            const Strides& get_window_movement_strides() const { return m_window_movement_strides; }
            const Strides& get_window_dilation_strides() const { return m_window_dilation_strides; }
            const CoordinateDiff& get_padding_below() const { return m_padding_below; }
            const CoordinateDiff& get_padding_above() const { return m_padding_above; }
            const Strides& get_data_dilation_strides() const { return m_data_dilation_strides; }
            std::shared_ptr<Node> get_bias() { return get_argument(2); }
            std::shared_ptr<Node> get_filters() { return get_argument(1); }
            std::shared_ptr<Node> get_data_batch() { return get_argument(0); }
            bool with_relu() const { return m_with_relu; }
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            void generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas) override;

        protected:
            Strides m_window_movement_strides;
            Strides m_window_dilation_strides;
            CoordinateDiff m_padding_below;
            CoordinateDiff m_padding_above;
            Strides m_data_dilation_strides;
            bool m_with_relu;
        };

        /// \brief Filters and bias backprop for batched convolution operation. Data backprop is
        /// the same as regular convolution backprop for data.
        class ConvolutionBiasBackpropFiltersBias : public Op
        {
        public:
            ConvolutionBiasBackpropFiltersBias(const std::shared_ptr<Node>& data_batch,
                                               const Shape& filters_shape,
                                               const Shape& bias_shape,
                                               const std::shared_ptr<Node>& output_delta,
                                               const Strides& window_movement_strides_forward,
                                               const Strides& window_dilation_strides_forward,
                                               const CoordinateDiff& padding_below_forward,
                                               const CoordinateDiff& padding_above_forward,
                                               const Strides& data_dilation_strides_forward);

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            /// \return The filters tensor shape.
            const Shape& get_filters_shape() const { return m_filters_shape; }
            /// \return The bias tensor shape.
            const Shape& get_bias_shape() const { return m_bias_shape; }
            /// \return The window movement strides from the forward prop.
            const Strides& get_window_movement_strides_forward() const
            {
                return m_window_movement_strides_forward;
            }
            /// \return The window dilation strides from the forward prop.
            const Strides& get_window_dilation_strides_forward() const
            {
                return m_window_dilation_strides_forward;
            }
            /// \return The padding-below sizes (possibly negative) from the forward prop.
            const CoordinateDiff& get_padding_below_forward() const
            {
                return m_padding_below_forward;
            }
            /// \return The padding-above sizes (possibly negative) from the forward prop.
            const CoordinateDiff& get_padding_above_forward() const
            {
                return m_padding_above_forward;
            }
            /// \return The data dilation strides from the forward prop.
            const Strides& get_data_dilation_strides_forward() const
            {
                return m_data_dilation_strides_forward;
            }

            /// \return The window movement strides for the backward prop.
            const Strides& get_window_movement_strides_backward() const
            {
                return m_window_movement_strides_backward;
            }
            /// \return The window dilation strides for the backward prop.
            const Strides& get_window_dilation_strides_backward() const
            {
                return m_window_dilation_strides_backward;
            }
            /// \return The padding-below sizes (possibly negative) for the backward prop.
            const CoordinateDiff& get_padding_below_backward() const
            {
                return m_padding_below_backward;
            }
            /// \return The padding-above sizes (possibly negative) for the backward prop.
            const CoordinateDiff& get_padding_above_backward() const
            {
                return m_padding_above_backward;
            }
            /// \return The data dilation strides for the backward prop.
            const Strides& get_data_dilation_strides_backward() const
            {
                return m_data_dilation_strides_backward;
            }

        protected:
            Shape m_filters_shape;
            Shape m_bias_shape;
            Strides m_window_movement_strides_forward;
            Strides m_window_dilation_strides_forward;
            CoordinateDiff m_padding_below_forward;
            CoordinateDiff m_padding_above_forward;
            Strides m_data_dilation_strides_forward;

            Strides m_window_movement_strides_backward;
            Strides m_window_dilation_strides_backward;
            CoordinateDiff m_padding_below_backward;
            CoordinateDiff m_padding_above_backward;
            Strides m_data_dilation_strides_backward;
        };

        class ConvolutionBiasAdd : public Op
        {
        public:
            ConvolutionBiasAdd(const std::shared_ptr<op::ConvolutionBias>& conv,
                               const std::shared_ptr<Node>& sum_input,
                               bool with_relu);

            ConvolutionBiasAdd(const std::shared_ptr<Node>& data_batch,
                               const std::shared_ptr<Node>& filters,
                               const std::shared_ptr<Node>& bias,
                               const std::shared_ptr<Node>& sum_input,
                               const Strides& window_movement_strides,
                               const Strides& window_dilation_strides,
                               const CoordinateDiff& padding_below,
                               const CoordinateDiff& padding_above,
                               const Strides& data_dilation_strides,
                               bool with_relu);

            const Strides& get_window_movement_strides() const { return m_window_movement_strides; }
            const Strides& get_window_dilation_strides() const { return m_window_dilation_strides; }
            const CoordinateDiff& get_padding_below() const { return m_padding_below; }
            const CoordinateDiff& get_padding_above() const { return m_padding_above; }
            const Strides& get_data_dilation_strides() const { return m_data_dilation_strides; }
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
        };

        namespace util
        {
            void validate_convbias_shapes(const Shape& data_shape,
                                          const Shape& filters_shape,
                                          const Shape& bias_shape);
        }
    }
}
