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
        /// \brief Batched average pooling operation, with optional padding and window stride.
        ///
        /// Average pooling takes as its input an image batch tensor of shape \f$(N,C,d_1,\dots,d_n)\f$ where \f$n > 0\f$, every \f$d_i > 0\f$, and where \f$N\f$ is
        /// the batch size, and \f$C > 0\f$ is the number of channels (sometimes called features). It also takes four parameters:
        ///
        /// 1. <i>(the window shape)</i> a size vector \f$(w_1,\dots,w_n)\f$ where every \f$w_i \le d_i\f$; and
        /// 2. <i>(the window movement strides, optional)</i> a vector of positive integers \f$(s_1,\dots,s_n)\f$.
        /// 3. <i>(the padding below, optional)</i> a vector of positive integers \f$(p_1,\dots,p_n)\f$.
        /// 4. <i>(the padding above, optional)</i> a vector of positive integers \f$(q_1,\dots,q_n)\f$.
        ///
        /// The output has the shape \f$(N,C,d'_1,\dots,d'_n)\f$, where \f$d'_n = \lceil \frac{p_i + d_i + q_i - w_i + 1}{s_i} \rceil\f$.
        ///
        /// *In the absence of padding*, given an input image batch tensor \f$T_\textit{in}\f$, the output tensor is defined by the equation
        ///
        /// \f[
        ///      T_\textit{out}[a,c,i_1,\dots,i_n] = \frac{\sum_{j_1 = s_1 i_1, \dots, j_n = s_n i_n}^{j_1 = s_1 i_1 + w_1 - 1, \dots, j_n = s_n i_n + w_n - 1} T_\textit{in}[a,c,j_1,\dots,j_n]}{\prod_{i=1}^n{w_n}}
        /// \f]
        ///
        /// *In the presence of padding*, we do not always want to divide by a reciprocal equal to the number of elements in the window, since some of the output points are
        /// determined by a window that is partly hanging beyond the edge of the tensor. In this case we can define the output via a few intermediate steps.
        ///
        /// First define the <i>sum tensor</i> \f$T_\textit{sum}\f$, with shape \f$(N,C,d'_1,\dots,d'_n)\f$, as follows.
        ///
        /// \f[
        ///      T_\textit{sum}[a,c,i_1,\dots,i_n] = \frac{\sum_{j_1 = s_1 i_1, \dots, j_n = s_n i_n}^{j_1 = s_1 i_1 + w_1 - 1, \dots, j_n = s_n i_n + w_n - 1} \textit{val}[a,c,j_1,\dots,j_n]}{\prod_{i=1}^n{w_n}}
        /// \f]
        ///
        /// where \f$\textit{val}[a,c,j_1,\dots,j_n] = T_\textit{in}[a,c,j_1,\dots,j_n]\f$ if for all \f$k\f$, \f$p_k \le j_k < p_k + d_k\f$; else \f$0\f$.
        ///
        /// Second, define the <i>divisor tensor</i> \f$T_\textit{div}\f$, with shape \f$(N,C,d'_1,\dots,d'_n)\f$, as follows.
        ///
        /// \f[
        ///      T_\textit{div}[a,c,i_1,\dots,i_n] = \frac{\sum_{j_1 = s_1 i_1, \dots, j_n = s_n i_n}^{j_1 = s_1 i_1 + w_1 - 1, \dots, j_n = s_n i_n + w_n - 1} \textit{val}[a,c,j_1,\dots,j_n]}{\prod_{i=1}^n{w_n}}
        /// \f]
        ///
        /// where \f$\textit{val}[a,c,j_1,\dots,j_n] = 1\f$ if for all \f$k\f$, \f$p_k \le j_k < p_k + d_k\f$; else \f$0\f$.
        ///
        /// Finally, define \f$T_\textit{out}\f$ as the result of elementwise dividing \f$T_\textit{sum}\f$ by \f$T_\textit{div}\f$.
        /// Note that at positions where \f$T_\textit{div}\f$ is zero, values may be infinity or nan. (This corresponds to a condition where the pooling window is completely
        /// out of bounds, encompassing no valid values.)
        class AvgPool : public RequiresTensorViewArgs
        {
        public:
            /// \brief Constructs a batched average pooling operation.
            ///
            /// \param arg The node producing the input image batch tensor.
            /// \param window_shape The window shape.
            /// \param window_movement_strides The window movement strides.
            /// \param padding_below The below-padding shape.
            /// \param padding_above The above-padding shape.
            AvgPool(const std::shared_ptr<Node>& arg,
                    const Shape& window_shape,
                    const Strides& window_movement_strides,
                    const Shape& padding_below,
                    const Shape& padding_above);

            /// \brief Constructs a batched, unpadded average pooling operation (i.e., all padding shapes are set to 0).
            ///
            /// \param arg The node producing the input image batch tensor.
            /// \param window_shape The window shape.
            /// \param window_movement_strides The window movement strides.
            AvgPool(const std::shared_ptr<Node>& arg,
                    const Shape& window_shape,
                    const Strides& window_movement_strides);

            /// \brief Constructs an unstrided batched convolution operation (i.e., all window movement strides are 1 and all padding shapes are set to 0).
            ///
            /// \param arg The node producing the input image batch tensor.
            /// \param window_shape The window shape.
            AvgPool(const std::shared_ptr<Node>& arg, const Shape& window_shape);

            virtual std::shared_ptr<Node> copy_with_new_args(
                const std::vector<std::shared_ptr<Node>>& new_args) const override
            {
                if (new_args.size() != 1)
                    throw ngraph_error("Incorrect number of new arguments");
                return std::make_shared<AvgPool>(new_args.at(0),
                                                 m_window_shape,
                                                 m_window_movement_strides,
                                                 m_padding_below,
                                                 m_padding_above);
            }

            /// \return The window shape.
            const Shape& get_window_shape() const { return m_window_shape; }
            /// \return The window movement strides.
            const Strides& get_window_movement_strides() const { return m_window_movement_strides; }
            /// \return The below-padding shape.
            const Shape& get_padding_below() const { return m_padding_below; }
            /// \return The above-padding shape.
            const Shape& get_padding_above() const { return m_padding_above; }
            /// \return The number of image channels.
            size_t get_channel_count() const { return m_channel_count; }
            /// \return The input image physical shape, not including padding.
            const Shape& get_input_image_physical_shape() const
            {
                return m_input_image_physical_shape;
            }
            /// \return The input image virtual shape, including padding.
            const Shape& get_input_image_virtual_shape() const
            {
                return m_input_image_virtual_shape;
            }
            /// \return The output image shape.
            const Shape& get_output_image_shape() const { return m_output_image_shape; }
            /// \return The batch size.
            size_t get_batch_size() const { return m_batch_size; }
            /// \return The number of image dimensions.
            size_t get_image_dimension_count() const { return m_image_dimension_count; }
            bool is_functionally_identical(const Node&) const override;

        protected:
            Shape m_window_shape;
            Strides m_window_movement_strides;
            Shape m_padding_below;
            Shape m_padding_above;

            size_t m_channel_count;
            Shape m_input_image_physical_shape;
            Shape m_input_image_virtual_shape;
            Shape m_output_image_shape;
            size_t m_batch_size;
            size_t m_image_dimension_count;
        };
    }
}
