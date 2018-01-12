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
        /// \brief Generic padding operation.
        ///
        /// Takes an input tensor of shape \f$(d_1,\dots,d_n)\f$ and pads by inserting a scalar \f$x\f$ supplied as input, in three possible ways:
        ///
        /// 1. <i>(exterior padding)</i> inserts copies of \f$x\f$ <i>below or above</i> the bounds of existing rows, columns, etc., or
        /// 2. <i>(interior padding)</i> inserts copies of \f$x\f$ <i>between</i> rows, columns, etc.,
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
        class Pad : public RequiresTensorViewArgs
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

            virtual std::shared_ptr<Node> copy_with_new_args(
                const std::vector<std::shared_ptr<Node>>& new_args) const override;
            /// \return The padding-below sizes.
            const Shape& get_padding_below() const { return m_padding_below; }
            /// \return The padding-above sizes.
            const Shape& get_padding_above() const { return m_padding_above; }
            /// \return The interior padding sizes.
            const Shape& get_padding_interior() const { return m_padding_interior; }
            bool is_functionally_identical(const Node&) const override;

        protected:
            Shape m_padding_below;
            Shape m_padding_above;
            Shape m_padding_interior;
        };
    }
}
