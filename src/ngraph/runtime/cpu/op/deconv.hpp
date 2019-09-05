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

#include "ngraph/op/convolution.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Deconvolution + Bias
        class DeconvolutionBias : public Op
        {
        public:
            static const std::string type_name;
            const std::string& description() const override { return type_name; }
            /// \brief Constructs a batched-convolution data batch-backprop operation.
            ///
            /// \param data_batch_shape The shape of the data batch from forward-prop.
            /// \param filters The node producing the filters from forward-prop.
            /// \param output_delta The node producing output delta.
            /// \param bias The node producing bias
            /// \param window_movement_strides_forward The window movement strides from
            ///        forward-prop.
            /// \param window_dilation_strides_forward The window dilation strides from
            ///        forward-prop.
            /// \param padding_below_forward The padding-below sizes from forward-prop.
            /// \param padding_above_forward The padding-above sizes from forward-prop.
            /// \param data_dilation_strides_forward The data dilation strides from forward-prop.
            /// \param with_relu Flag indicating to add relu or not
            DeconvolutionBias(const Shape& data_batch_shape,
                              const Output<Node>& filters,
                              const Output<Node>& output_delta,
                              const Output<Node>& bias,
                              const Strides& window_movement_strides_forward,
                              const Strides& window_dilation_strides_forward,
                              const CoordinateDiff& padding_below_forward,
                              const CoordinateDiff& padding_above_forward,
                              const Strides& data_dilation_strides_forward,
                              const bool with_relu);

            void validate_and_infer_types() override;

            void generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas) override;
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            /// \return The data batch shape.
            const Shape& get_data_batch_shape() const { return m_data_batch_shape; }
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
            /// \return The input data dilation strides from the forward prop.
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
            /// \return The input data dilation strides for the backward prop.
            const Strides& get_data_dilation_strides_backward() const
            {
                return m_data_dilation_strides_backward;
            }
            bool with_relu() const { return m_with_relu; }
        protected:
            Shape m_data_batch_shape;
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
            bool m_with_relu;
        };
    }
}
