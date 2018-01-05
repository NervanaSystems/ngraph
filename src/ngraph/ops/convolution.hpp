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
        ///
        /// Convolution takes two inputs:
        ///
        /// 1. <i>(the image batch)</i> a tensor of shape \f$(N,C_\textit{in},d_1,\dots,d_n)\f$ where \f$n > 0\f$, every \f$d_i > 0\f$, and where \f$N\f$ is the batch size
        ///    (number of images) and \f$C_\textit{in} > 0\f$ is the number of input channels (sometimes called features); and
        /// 2. <i>(the filters)</i> a tensor of shape \f$(C_\textit{out},C_\textit{in},d^f_1,\dots,d^f_n)\f$, where \f$C_\textit{out} > 0\f$ is the number of output channels
        ///    (sometimes called features) and \f$(d^f_1,\dots,d^f_n)\f$ are the filter dimensions. It is required that for all \f$i\f$, \f$0 < l_i(d^f_i - 1) + 1 \le d_i\f$.
        ///    (See below for the definition of the dilation \f$l_i\f$);
        ///
        /// and four optional parameters:
        ///
        /// 3. <i>(the window movement strides)</i> a vector of positive integers \f$(s_1,\dots,s_n)\f$ (default is all ones),
        /// 4. <i>(the window dilation strides)</i> a vector of positive integers \f$(l_1,\dots,l_n)\f$ (default is all ones),
        /// 5. <i>(the padding below)</i> a vector of non-negative integers \f$(p_1,\dots,p_n)\f$ (default is all zeros), and
        /// 6. <i>(the padding above)</i> a vector of non-negative integers \f$(q_1,\dots,q_n)\f$ (default is all zeros).
        ///
        /// The output has the shape \f$(N,C_\textit{out},d'_1,\dots,d'_n)\f$, where \f$d'_n = \lceil \frac{d_i + p_i + q_i - l_i(d^f_i - 1)}{s_i} \rceil\f$.
        ///
        /// Given an input image batch tensor \f$T_\textit{in}\f$, first define the <i>padded input tensor</i> \f$T_\textit{pad}\f$, with shape \f$(N,C_\textit{in},d_1+p_1+q+1,\dots,d_n+p_n+q_n)\f$, as follows:
        ///
        /// \f[
        ///      T_\textit{pad}[a,c,i_1,\dots,i_n] = T[a,c,i_1 - p_1,\dots,i_n - p_n] \text{ if for all }k, p_k \le i_k \lt p_k + d_k, \text{ else } 0
        /// \f]
        ///
        /// then, given an input filter tensor \f$T_\textit{filt}\f$, the output tensor \f$T_\textit{out}\f$ is defined by the equation.
        ///
        /// \f[
        ///      T_\textit{out}[a,c_\textit{out},i_1,\dots,i_n] = \sum_{c_\textit{in}=0,j_1=0,\dots,j_n=0}^{c_\textit{in}=C_\textit{in}-1,j_1=d^f_1-1,\dots,j_n=d^f_n-1} (T_\textit{filt}[c_\textit{out},c_\textit{in},j_1,\dots,j_n] \cdot T_\textit{pad}[a,c_\textit{in},s_1i_1+l_1j_1,\dots,s_ni_n+l_nj_n])
        /// \f]
        ///
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
            Convolution(const std::shared_ptr<Node>& image_batch,
                        const std::shared_ptr<Node>& filters,
                        const Strides& window_movement_strides,
                        const Strides& window_dilation_strides,
                        const Shape& padding_below,
                        const Shape& padding_above);

            /// \brief Constructs a batched convolution operation with no padding (i.e., padding above and below are 0 everywhere).
            ///
            /// \param image_batch The node producing the input image batch tensor.
            /// \param filters The node producing the filters tensor.
            /// \param window_movement_strides The window movement strides.
            /// \param window_dilation_strides The window dilation strides.
            Convolution(const std::shared_ptr<Node>& image_batch,
                        const std::shared_ptr<Node>& filters,
                        const Strides& window_movement_strides,
                        const Strides& window_dilation_strides);

            /// \brief Constructs a batched convolution operation with no window dilation (i.e., all dilation strides are 1).
            ///
            /// \param image_batch The node producing the input image batch tensor.
            /// \param filters The node producing the filters tensor.
            /// \param window_movement_strides The window movement strides.
            Convolution(const std::shared_ptr<Node>& image_batch,
                        const std::shared_ptr<Node>& filters,
                        const Strides& window_movement_strides);

            /// \brief Constructs a batched convolution operation with no window dilation or movement stride (i.e., all dilation and movement strides are 1).
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
            /// \return The padding-below sizes.
            const Shape& get_padding_below() const { return m_padding_below; }
            /// \return The padding-above sizes.
            const Strides& get_padding_above() const { return m_padding_above; }
            /// \return The number of input channels.
            size_t get_input_channel_count() const { return m_input_channel_count; }
            /// \return The number of output channels.
            size_t get_output_channel_count() const { return m_output_channel_count; }
            /// \return The input image shape, not including padding.
            const Shape& get_input_image_shape() const { return m_input_image_shape; }
            /// \return The input image shape, including padding.
            const Shape& get_padded_input_image_shape() const { return m_padded_input_image_shape; }
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
        protected:
            Strides m_window_movement_strides;
            Strides m_window_dilation_strides;
            Shape m_padding_below;
            Shape m_padding_above;

            // TODO: Some of these values should probably be computed dynamically rather than stored here.
            size_t m_input_channel_count;
            size_t m_output_channel_count;
            Shape m_input_image_shape;
            Shape m_padded_input_image_shape;
            Shape m_output_image_shape;
            Shape m_window_physical_shape;
            Shape m_window_virtual_shape;
            size_t m_batch_size;
            size_t m_image_dimension_count;
        };
    }
}
