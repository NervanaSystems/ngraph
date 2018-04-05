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

#include "ngraph/op/util/requires_tensor_view_args.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Batched max pooling operation, with optional padding and window stride.
        ///
        /// (TODO: add an account of the optional padding to this comment.)
        ///
        /// Max pooling takes as its input a data batch tensor of shape \f$(N,C,d_1,\dots,d_n)\f$ where \f$n > 0\f$, every \f$d_i > 0\f$, and where \f$N\f$ is
        /// the batch size, and \f$C > 0\f$ is the number of channels (sometimes called features). The dimensions \f$(d_1,\dots,d_n)\f$ correspond to the shape of
        /// an \f$n\f$-dimensional data item in a batch. For example, where \f$n=2\f$, the data may represent a two-dimensional image. It also takes two parameters:
        ///
        /// 1. <i>(the window shape)</i> a size vector \f$(w_1,\dots,w_n)\f$ where every \f$w_i \le d_i\f$; and
        /// 2. <i>(the window movement strides, optional)</i> a vector of positive integers \f$(s_1,\dots,s_n)\f$.
        ///
        /// The output has the shape \f$(N,C,d'_1,\dots,d'_n)\f$, where \f$d'_n = \lceil \frac{d_i - w_i + 1}{s_i} \rceil\f$.
        ///
        /// Given an input data batch tensor \f$T_\textit{in}\f$, the output tensor is defined by the equation
        ///
        /// \f[
        ///      T_\textit{out}[a,c,i_1,\dots,i_n] = \max_{j_1 = s_1 i_1, \dots, j_n = s_n i_n}^{j_1 = s_1 i_1 + w_1 - 1, \dots, j_n = s_n i_n + w_n - 1} (T_\textit{in}[a,c,j_1,\dots,j_n])
        /// \f]
        ///
        class MaxPool : public util::RequiresTensorViewArgs
        {
        public:
            /// \brief Constructs a batched max pooling operation.
            ///
            /// \param arg The node producing the input data batch tensor.
            /// \param window_shape The window shape.
            /// \param window_movement_strides The window movement strides.
            /// \param padding_below The below-padding shape.
            /// \param padding_above The above-padding shape.
            MaxPool(const std::shared_ptr<Node>& arg,
                    const Shape& window_shape,
                    const Strides& window_movement_strides,
                    const Shape& padding_below,
                    const Shape& padding_above);

            /// \brief Constructs a batched, unpadded max pooling operation (i.e., all padding shapes are set to 0).
            ///
            /// \param arg The node producing the input data batch tensor.
            /// \param window_shape The window shape.
            /// \param window_movement_strides The window movement strides.
            MaxPool(const std::shared_ptr<Node>& arg,
                    const Shape& window_shape,
                    const Strides& window_movement_strides);

            /// \brief Constructs an unstrided batched max pooling operation (i.e., all window movement strides are 1 and all padding shapes are set to 0).
            ///
            /// \param arg The node producing the input data batch tensor.
            /// \param window_shape The window shape.
            MaxPool(const std::shared_ptr<Node>& arg, const Shape& window_shape);

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            /// \return The window shape.
            const Shape& get_window_shape() const { return m_window_shape; }
            /// \return The window movement strides.
            const Strides& get_window_movement_strides() const { return m_window_movement_strides; }
            /// \return The below-padding shape.
            const Shape& get_padding_below() const { return m_padding_below; }
            /// \return The above-padding shape.
            const Shape& get_padding_above() const { return m_padding_above; }
        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;

            Shape m_window_shape;
            Strides m_window_movement_strides;
            Shape m_padding_below;
            Shape m_padding_above;
        };

        class MaxPoolBackprop : public util::RequiresTensorViewArgs
        {
        public:
            MaxPoolBackprop(const std::shared_ptr<Node>& arg_forward,
                            const std::shared_ptr<Node>& delta,
                            const Shape& window_shape,
                            const Strides& window_movement_strides,
                            const Shape& padding_below,
                            const Shape& padding_above,
                            const std::shared_ptr<op::MaxPool>& forward_op = nullptr);

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            const Shape& get_window_shape() const { return m_window_shape; }
            const Strides& get_window_movement_strides() const { return m_window_movement_strides; }
            const Shape& get_padding_below() const { return m_padding_below; }
            const Shape& get_padding_above() const { return m_padding_above; }
            /// \return A pointer to the corresponding `MaxPool` forward prop op. This may be
            ///         `nullptr` if no such pointer was provided at construction time, or if the
            ///         forward op has been freed due to graph rewriting.
            std::shared_ptr<op::MaxPool> get_forward_op() const;

        protected:
            Shape m_window_shape;
            Strides m_window_movement_strides;
            Shape m_padding_below;
            Shape m_padding_above;
            std::weak_ptr<op::MaxPool> m_forward_op;
        };
    }
}
