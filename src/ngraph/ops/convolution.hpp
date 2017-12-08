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
        /// \brief Batched convolution op.
        class Convolution : public RequiresTensorViewArgs
        {
        public:
            /// \brief Constructs XXX
            ///
            /// \param arg0 XXX
            /// \param arg1 XXX
            /// \param window_movement_strides XXX
            /// \param window_dilation_strides XXX
            Convolution(const std::shared_ptr<Node>& arg0,
                        const std::shared_ptr<Node>& arg1,
                        const Strides& window_movement_strides,
                        const Strides& window_dilation_strides);

            // Undilated.
            Convolution(const std::shared_ptr<Node>& arg0,
                        const std::shared_ptr<Node>& arg1,
                        const Strides& window_movement_strides);

            // Undilated, unit stride.
            Convolution(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1);

            virtual std::shared_ptr<Node> copy_with_new_args(
                const std::vector<std::shared_ptr<Node>>& new_args) const override
            {
                if (new_args.size() != 2)
                    throw ngraph_error("Incorrect number of new arguments");
                return std::make_shared<Convolution>(new_args.at(0),
                                                     new_args.at(1),
                                                     m_window_movement_strides,
                                                     m_window_dilation_strides);
            }

            /// \return The window movement strides.
            const Strides& get_window_movement_strides() const { return m_window_movement_strides; }
            /// \return The window dilation strides.
            const Strides& get_window_dilation_strides() const { return m_window_dilation_strides; }
            /// \return The number of input channels.
            size_t get_n_input_channels() const { return m_n_input_channels; }
            /// \return The number of output channels.
            size_t get_n_output_channels() const { return m_n_output_channels; }
            /// \return The input image shape.
            Shape get_input_image_shape() const { return m_input_image_shape; }
            /// \return The output image shape.
            Shape get_output_image_shape() const { return m_output_image_shape; }
            /// \return The physical window shape.
            Shape get_window_physical_shape() const { return m_window_physical_shape; }
            /// \return The virtual window shape.
            Shape get_window_virtual_shape() const { return m_window_virtual_shape; }
            /// \return The batch size.
            size_t get_batch_size() const { return m_batch_size; }
            /// \return The number of image dimensions.
            size_t get_n_image_dimensions() const { return m_n_image_dimensions; }
        protected:
            Strides m_window_movement_strides;
            Strides m_window_dilation_strides;

            size_t m_n_input_channels;
            size_t m_n_output_channels;
            Shape m_input_image_shape;
            Shape m_output_image_shape;
            Shape m_window_physical_shape;
            Shape m_window_virtual_shape;
            size_t m_batch_size;
            size_t m_n_image_dimensions;
        };
    }
}
