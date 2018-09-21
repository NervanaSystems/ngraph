//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Generic constant-padding operation.
        class Pad : public Op
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
            /// \return The default value for Pad.
            virtual std::shared_ptr<Node> get_default_value() const override;

        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;
            Shape m_padding_below;
            Shape m_padding_above;
            Shape m_padding_interior;
        };
    }
}
