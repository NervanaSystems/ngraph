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
        /// \brief Generic constant-padding operation.
        ///
        /// Takes an input tensor of shape \f$(d_1,\dots,d_n)\f$ and pads by inserting a scalar \f$x\f$ supplied as input, in three possible ways:
        ///
        /// 1. <i>(exterior padding)</i> inserts copies of \f$x\f$ <i>below or above</i> the bounds of existing rows, columns, etc.,
        /// 2. <i>(interior padding)</i> inserts copies of \f$x\f$ <i>between</i> rows, columns, etc., or
        /// 3. both of the above.
        ///
        /// The number and position of elements to be inserted along a given axis is determined by three parameters:
        ///
        /// 1. <i>(the padding-below sizes)</i> a vector of non-negative integers \f$(p_1,\dots,p_n)\f$,
        /// 2. <i>(the padding-above sizes)</i> a vector of non-negative integers \f$(q_1,\dots,q_n)\f$, and
        /// 3. <i>(the interior padding sizes)</i> a vector of non-negative integers \f$(r_1,\dots,r_n)\f$.
        ///
        /// The output tensor will have the shape \f$(d'_1,\dots,d'_n)\f$ where \f$d'_i = p_i + (d_i - 1)(r_i + 1) + 1 + q_i\f$ if \f$d_i > 0\f$, and \f$d'_i = p_i + q_i\f$ if \f$d_i = 0\f$.
        ///
        /// Example: given a 3x3 tensor, with interior-padding sizes of `{1,2}`, padding-below of `{1,2}`, padding-above of `{1,0}`, and a pad-value of `42`, we obtain:
        ///
        /// ```
        ///           42 42 42 42 42 42 42 42 42
        ///           42 42  1 42 42  2 42 42  3
        /// 1 2 3     42 42 42 42 42 42 42 42 42
        /// 4 5 6 --> 42 42  4 42 42  5 42 42  6
        /// 7 8 9     42 42 42 42 42 42 42 42 42
        ///           42 42  7 42 42  8 42 42  9
        ///           42 42 42 42 42 42 42 42 42
        /// ```
        ///
        /// In other words we have inserted one new row between each pair of adjacent rows, two new columns between each pair of adjacent columns, one new row at
        /// the top and two new columns on the left, and one new row at the bottom and zero new columns on the right; then filled the new rows and columns with `42`.
        ///
        /// (Note that `below` and `above` here refer respectively to lower- or higher-numbered coordinate indices, and numbering starts at the upper-left corner;
        /// thus inserting a row "below" actually inserts it at the "top" of the matrix.)
        ///
        class Pad : public util::RequiresTensorViewArgs
        {
        public:
            /// \brief Constructs a generic padding operation.
            ///
            /// \param arg The node producing input tensor to be padded.
            /// \param arg_pad_value The node producing the scalar value to be inserted for padding.
            /// \param padding_below The padding-below widths.
            /// \param padding_above The padding-above widths.
            /// \param padding_interior The interior-padding widths.
            Pad(const std::shared_ptr<Node>& arg,
                const std::shared_ptr<Node>& arg_pad_value,
                const Shape& padding_below,
                const Shape& padding_above,
                const Shape& padding_interior);

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;
            /// \return The padding-below sizes.
            const Shape& get_padding_below() const { return m_padding_below; }
            /// \return The padding-above sizes.
            const Shape& get_padding_above() const { return m_padding_above; }
            /// \return The interior padding sizes.
            const Shape& get_padding_interior() const { return m_padding_interior; }
        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;
            Shape m_padding_below;
            Shape m_padding_above;
            Shape m_padding_interior;
        };
    }
}
