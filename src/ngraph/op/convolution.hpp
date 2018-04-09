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

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/util/requires_tensor_view_args.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Batched convolution operation, with optional window dilation and stride.
        ///
        class Convolution : public util::RequiresTensorViewArgs
        {
        public:
            /// \brief Constructs a batched convolution operation.
            ///
            /// \param data_batch The node producing the input data batch tensor.<br>
            /// `[N, C_IN, D1, ... Df]`
            /// \param filters The node producing the filters tensor.<br>
            /// `[C_OUT, C_IN, F1, ... Ff]`
            /// \param window_movement_strides The window movement strides.<br>
            /// `[f]`
            /// \param window_dilation_strides The window dilation strides.<br>
            /// `[f]`
            /// \param padding_below The padding-below sizes.<br>
            /// `[f]`
            /// \param padding_above The padding-above sizes.<br>
            /// `[f]`
            /// \param data_dilation_strides The data dilation strides.<br>
            /// `[f]`
            ///
            /// Output `[N, C_OUT, R1, ... Rf]`
            ///
            Convolution(const std::shared_ptr<Node>& data_batch,
                        const std::shared_ptr<Node>& filters,
                        const Strides& window_movement_strides,
                        const Strides& window_dilation_strides,
                        const CoordinateDiff& padding_below,
                        const CoordinateDiff& padding_above,
                        const Strides& data_dilation_strides);

            /// \brief Constructs a batched convolution operation with no data dilation (i.e., all data dilation strides are 1).
            ///
            /// \param data_batch The node producing the input data batch tensor.<br>
            /// `[N, C_IN, D1, ... Df]`
            /// \param filters The node producing the filters tensor.<br>
            /// `[C_OUT, C_IN, F1, ... Ff]`
            /// \param window_movement_strides The window movement strides.<br>
            /// `[f]`
            /// \param window_dilation_strides The window dilation strides.<br>
            /// `[f]`
            /// \param padding_below The padding-below sizes.<br>
            /// `[f]`
            /// \param padding_above The padding-above sizes.<br>
            /// `[f]`
            ///
            /// Output `[N, C_OUT, R1, ... Rf]`
            ///
            Convolution(const std::shared_ptr<Node>& data_batch,
                        const std::shared_ptr<Node>& filters,
                        const Strides& window_movement_strides,
                        const Strides& window_dilation_strides,
                        const CoordinateDiff& padding_below,
                        const CoordinateDiff& padding_above);

            /// \brief Constructs a batched convolution operation with no padding or data dilation (i.e., padding above and below are 0 everywhere, and all data dilation strides are 1).
            ///
            /// \param data_batch The node producing the input data batch tensor.<br>
            /// `[N, C_IN, D1, ... Df]`
            /// \param filters The node producing the filters tensor.<br>
            /// `[C_OUT, C_IN, F1, ... Ff]`
            /// \param window_movement_strides The window movement strides.<br>
            /// `[f]`
            /// \param window_dilation_strides The window dilation strides.<br>
            /// `[f]`
            ///
            /// Output `[N, C_OUT, R1, ... Rf]`
            ///
            Convolution(const std::shared_ptr<Node>& data_batch,
                        const std::shared_ptr<Node>& filters,
                        const Strides& window_movement_strides,
                        const Strides& window_dilation_strides);

            /// \brief Constructs a batched convolution operation with no window dilation, padding, or data dilation (i.e., padding above and below are 0 everywhere, and all window/data dilation strides are 1).
            ///
            /// \param data_batch The node producing the input data batch tensor.<br>
            /// `[N, C_IN, D1, ... Df]`
            /// \param filters The node producing the filters tensor.<br>
            /// `[C_OUT, C_IN, F1, ... Ff]`
            /// \param window_movement_strides The window movement strides.<br>
            /// `[f]`
            ///
            /// Output `[N, C_OUT, R1, ... Rf]`
            ///
            Convolution(const std::shared_ptr<Node>& data_batch,
                        const std::shared_ptr<Node>& filters,
                        const Strides& window_movement_strides);

            /// \brief Constructs a batched convolution operation with no window dilation or movement stride (i.e., padding above and below are 0 everywhere, and all window/data dilation strides and window movement strides are 1).
            ///
            /// \param data_batch The node producing the input data batch tensor.<br>
            /// `[N, C_IN, D1, ... Df]`
            /// \param filters The node producing the filters tensor.<br>
            /// `[C_OUT, C_IN, F1, ... Ff]`
            ///
            /// Output `[N, C_OUT, R1, ... Rf]`
            ///
            Convolution(const std::shared_ptr<Node>& data_batch,
                        const std::shared_ptr<Node>& filters);

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;
            void generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas) override;

            /// \return The window movement strides.
            const Strides& get_window_movement_strides() const { return m_window_movement_strides; }
            /// \return The window dilation strides.
            const Strides& get_window_dilation_strides() const { return m_window_dilation_strides; }
            /// \return The padding-below sizes (possibly negative).
            const CoordinateDiff& get_padding_below() const { return m_padding_below; }
            /// \return The padding-above sizes (possibly negative).
            const CoordinateDiff& get_padding_above() const { return m_padding_above; }
            /// \return The input data dilation strides.
            const Strides& get_data_dilation_strides() const { return m_data_dilation_strides; }
        protected:
            Strides m_window_movement_strides;
            Strides m_window_dilation_strides;
            CoordinateDiff m_padding_below;
            CoordinateDiff m_padding_above;
            Strides m_data_dilation_strides;

        private:
            static Strides default_strides(const std::shared_ptr<Node>& data_batch);
            static CoordinateDiff default_padding(const std::shared_ptr<Node>& data_batch);
        };

        /// \brief Data batch backprop for batched convolution operation.
        class ConvolutionBackpropData : public util::RequiresTensorViewArgs
        {
        public:
            /// \brief Constructs a batched-convolution data batch-backprop operation.
            ///
            /// \param data_batch_shape The shape of the data batch from forward-prop.
            /// \param filters The node producing the filters from forward-prop.
            /// \param output_delta The node producing output delta.
            /// \param window_movement_strides_forward The window movement strides from forward-prop.
            /// \param window_dilation_strides_forward The window dilation strides from forward-prop.
            /// \param padding_below_forward The padding-below sizes from forward-prop.
            /// \param padding_above_forward The padding-above sizes from forward-prop.
            /// \param data_dilation_strides_forward The data dilation strides from forward-prop.
            ConvolutionBackpropData(const Shape& data_batch_shape,
                                    const std::shared_ptr<Node>& filters,
                                    const std::shared_ptr<Node>& output_delta,
                                    const Strides& window_movement_strides_forward,
                                    const Strides& window_dilation_strides_forward,
                                    const CoordinateDiff& padding_below_forward,
                                    const CoordinateDiff& padding_above_forward,
                                    const Strides& data_dilation_strides_forward);

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
        };

        /// \brief Filters backprop for batched convolution operation.
        class ConvolutionBackpropFilters : public util::RequiresTensorViewArgs
        {
        public:
            /// \brief Constructs a batched-convolution filter-backprop operation.
            ///
            /// \param data_batch The tensor producing the data batch from forward-prop.
            /// \param filters_shape The shape of the filters from forward-prop.
            /// \param output_delta The node producing output delta.
            /// \param window_movement_strides_forward The window movement strides from forward-prop.
            /// \param window_dilation_strides_forward The window dilation strides from forward-prop.
            /// \param padding_below_forward The padding-below sizes from forward-prop.
            /// \param padding_above_forward The padding-above sizes from forward-prop.
            /// \param data_dilation_strides_forward The data dilation strides from forward-prop.
            ConvolutionBackpropFilters(const std::shared_ptr<Node>& data_batch,
                                       const Shape& filters_shape,
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

        namespace util
        {
            Shape infer_convolution_output_shape(const Shape& data_batch_shape,
                                                 const Shape& filters_shape,
                                                 const Strides& window_movement_strides,
                                                 const Strides& window_dilation_strides,
                                                 const CoordinateDiff& padding_below,
                                                 const CoordinateDiff& padding_above,
                                                 const Strides& data_dilation_strides,
                                                 size_t batch_axis_data,
                                                 size_t input_channel_axis_data,
                                                 size_t input_channel_axis_filters,
                                                 size_t output_channel_axis_filters,
                                                 size_t batch_axis_result,
                                                 size_t output_channel_axis_result,
                                                 const std::string& error_prefix);
        }
    }
}
