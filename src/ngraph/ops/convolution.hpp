// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#pragma once

#include "ngraph/ops/op.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Batched convolution operation, with optional window dilation and stride.
        class Convolution : public RequiresTensorViewArgs
        {
        public:
            /// \brief Constructs a batched convolution operation.
            ///
            /// \param image_batch The node producing the input image batch tensor.
            /// \param filters The node producing the filters tensor.
            /// \param window_movement_strides The window movement strides.
            /// \param window_dilation_strides The window dilation strides.
            /// \param padding_below The padding-below sizes.
            /// \param padding_above The padding-above sizes.
            /// \param image_dilation_strides The image dilation strides.
            Convolution(const std::shared_ptr<Node>& image_batch,
                        const std::shared_ptr<Node>& filters,
                        const Strides& window_movement_strides,
                        const Strides& window_dilation_strides,
                        const CoordinateDiff& padding_below,
                        const CoordinateDiff& padding_above,
                        const Strides& image_dilation_strides);

            /// \brief Constructs a batched convolution operation with no image dilation (i.e., all image dilation strides are 1).
            ///
            /// \param image_batch The node producing the input image batch tensor.
            /// \param filters The node producing the filters tensor.
            /// \param window_movement_strides The window movement strides.
            /// \param window_dilation_strides The window dilation strides.
            /// \param padding_below The padding-below sizes.
            /// \param padding_above The padding-above sizes.
            Convolution(const std::shared_ptr<Node>& image_batch,
                        const std::shared_ptr<Node>& filters,
                        const Strides& window_movement_strides,
                        const Strides& window_dilation_strides,
                        const CoordinateDiff& padding_below,
                        const CoordinateDiff& padding_above);

            /// \brief Constructs a batched convolution operation with no padding or image dilation (i.e., padding above and below are 0 everywhere, and all image dilation strides are 1).
            ///
            /// \param image_batch The node producing the input image batch tensor.
            /// \param filters The node producing the filters tensor.
            /// \param window_movement_strides The window movement strides.
            /// \param window_dilation_strides The window dilation strides.
            Convolution(const std::shared_ptr<Node>& image_batch,
                        const std::shared_ptr<Node>& filters,
                        const Strides& window_movement_strides,
                        const Strides& window_dilation_strides);

            /// \brief Constructs a batched convolution operation with no window dilation, padding, or image dilation (i.e., padding above and below are 0 everywhere, and all window/image dilation strides are 1).
            ///
            /// \param image_batch The node producing the input image batch tensor.
            /// \param filters The node producing the filters tensor.
            /// \param window_movement_strides The window movement strides.
            Convolution(const std::shared_ptr<Node>& image_batch,
                        const std::shared_ptr<Node>& filters,
                        const Strides& window_movement_strides);

            /// \brief Constructs a batched convolution operation with no window dilation or movement stride (i.e., padding above and below are 0 everywhere, and all window/image dilation strides and window movement strides are 1).
            ///
            /// \param image_batch The node producing the input image batch tensor.
            /// \param filters The node producing the filters tensor.
            Convolution(const std::shared_ptr<Node>& image_batch,
                        const std::shared_ptr<Node>& filters);

            virtual std::shared_ptr<Node> copy_with_new_args(
                const std::vector<std::shared_ptr<Node>>& new_args) const override;

            /// \return The window movement strides.
            const Strides& get_window_movement_strides() const { return m_window_movement_strides; }
            /// \return The window dilation strides.
            const Strides& get_window_dilation_strides() const { return m_window_dilation_strides; }
            /// \return The padding-below sizes (possibly negative).
            const CoordinateDiff& get_padding_below() const { return m_padding_below; }
            /// \return The padding-above sizes (possibly negative).
            const CoordinateDiff& get_padding_above() const { return m_padding_above; }
            /// \return The input image dilation strides.
            const Strides& get_image_dilation_strides() const { return m_image_dilation_strides; }
            /// \return The number of input channels.
            size_t get_input_channel_count() const { return m_input_channel_count; }
            /// \return The number of output channels.
            size_t get_output_channel_count() const { return m_output_channel_count; }
            /// \return The physical input image shape, not including padding and dilation.
            const Shape& get_input_image_physical_shape() const
            {
                return m_input_image_physical_shape;
            }
            /// \return The virtual input image shape, including padding and dilation.
            const Shape& get_input_image_virtual_shape() const
            {
                return m_input_image_virtual_shape;
            }
            /// \return The output image shape.
            const Shape& get_output_image_shape() const { return m_output_image_shape; }
            /// \return The physical window shape.
            const Shape& get_window_physical_shape() const { return m_window_physical_shape; }
            /// \return The virtual window shape.
            const Shape& get_window_virtual_shape() const { return m_window_virtual_shape; }
            /// \return The batch size.
            size_t get_batch_size() const { return m_batch_size; }
            /// \return The number of image dimensions.
            size_t get_image_dimension_count() const { return m_image_dimension_count; }
            bool is_functionally_identical(const Node&) const override;

        protected:
            Strides m_window_movement_strides;
            Strides m_window_dilation_strides;
            CoordinateDiff m_padding_below;
            CoordinateDiff m_padding_above;
            Strides m_image_dilation_strides;

            // TODO: Some or all of these values should probably be computed dynamically rather than stored here.
            size_t m_input_channel_count;
            size_t m_output_channel_count;
            Shape m_input_image_physical_shape;
            Shape m_input_image_virtual_shape;
            Shape m_output_image_shape;
            Shape m_window_physical_shape;
            Shape m_window_virtual_shape;
            size_t m_batch_size;
            size_t m_image_dimension_count;

        private:
            static Strides default_strides(const std::shared_ptr<Node>& image_batch);
            static CoordinateDiff default_padding(const std::shared_ptr<Node>& image_batch);
        };
    }
}
