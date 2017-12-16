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
        /// \brief Batched max pooling operation, with optional window stride.
        ///
        /// Max pooling takes as its input an image batch tensor of shape \f$(N,C,d_1,\dots,d_n)\f$ where \f$n > 0\f$, every \f$d_i > 0\f$, and where \f$N\f$ is
        /// the batch size, and \f$C > 0\f$ is the number of channels (sometimes called features). It also takes two parameters:
        ///
        /// 1. <i>(the window shape)</i> a size vector \f$(w_1,\dots,w_n)\f$ where every \f$w_i \le d_i\f$; and
        /// 2. <i>(the window movement strides, optional)</i> a vector of positive integers \f$(s_1,\dots,s_n)\f$.
        ///
        /// The output has the shape \f$(N,C,d'_1,\dots,d'_n)\f$, where \f$d'_n = \lceil \frac{d_i - w_i + 1}{s_i} \rceil\f$.
        ///
        /// Given an input image batch tensor \f$T_\textit{in}\f$, the output tensor is defined by the equation
        ///
        /// \f[
        ///      T_\textit{out}[a,c,i_1,\dots,i_n] = \max_{j_1 = i_1, \dots, j_n = i_n}^{j_1 = i_1 + w_1 - 1, \dots, j_n = i_n + w_n - 1} (T_\textit{in}[a,c,j_1,\dots,j_n])
        /// \f]
        ///
        class MaxPool : public RequiresTensorViewArgs
        {
        public:
            /// \brief Constructs a batched max pooling operation.
            ///
            /// \param arg The node producing the input image batch tensor.
            /// \param window_shape The window shape.
            /// \param window_movement_strides The window movement strides.
            MaxPool(const std::shared_ptr<Node>& arg,
                    const Shape& window_shape,
                    const Strides& window_movement_strides);

            /// \brief Constructs an unstrided batched convolution operation (i.e., all window movement strides are 1).
            ///
            /// \param arg The node producing the input image batch tensor.
            /// \param window_shape The window shape.
            MaxPool(const std::shared_ptr<Node>& arg, const Shape& window_shape);

            virtual std::shared_ptr<Node> copy_with_new_args(
                const std::vector<std::shared_ptr<Node>>& new_args) const override
            {
                if (new_args.size() != 1)
                    throw ngraph_error("Incorrect number of new arguments");
                return std::make_shared<MaxPool>(
                    new_args.at(0), m_window_shape, m_window_movement_strides);
            }

            /// \return The window shape.
            Shape get_window_shape() const { return m_window_shape; }
            /// \return The window movement strides.
            const Strides& get_window_movement_strides() const { return m_window_movement_strides; }
            /// \return The number of image channels.
            size_t get_channel_count() const { return m_channel_count; }
            /// \return The input image shape.
            Shape get_input_image_shape() const { return m_input_image_shape; }
            /// \return The output image shape.
            Shape get_output_image_shape() const { return m_output_image_shape; }
            /// \return The batch size.
            size_t get_batch_size() const { return m_batch_size; }
            /// \return The number of image dimensions.
            size_t get_image_dimension_count() const { return m_image_dimension_count; }
        protected:
            Shape m_window_shape;
            Strides m_window_movement_strides;

            size_t m_channel_count;
            Shape m_input_image_shape;
            Shape m_output_image_shape;
            size_t m_batch_size;
            size_t m_image_dimension_count;
        };
    }
}
