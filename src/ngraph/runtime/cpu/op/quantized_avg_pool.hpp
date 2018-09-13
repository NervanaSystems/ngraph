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

#include "ngraph/graph_util.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Batched average pooling operation, with optional padding and window stride.
        ///
        class QuantizedAvgPool : public Op
        {
        public:
            /// \brief Constructs a batched average pooling operation.
            ///
            /// \param arg The node producing the input data batch tensor.<br>
            /// `[d1, dn]`
            /// \param window_shape The window shape.<br>
            /// `[n]`
            /// \param window_movement_strides The window movement strides.<br>
            /// `[n]`
            /// \param padding_below The below-padding shape.<br>
            /// `[n]`
            /// \param padding_above The above-padding shape.<br>
            /// `[n]`
            /// \param include_padding_in_avg_computation If true then averages include padding
            ///  elements, each treated as the number zero.  If false, padding elements are entirely
            ///  ignored when computing averages.
            QuantizedAvgPool(const std::shared_ptr<Node>& arg,
                             const Shape& window_shape,
                             const Strides& window_movement_strides,
                             const Shape& padding_below,
                             const Shape& padding_above,
                             bool include_padding_in_avg_computation,
                             const std::shared_ptr<Node> min,
                             const std::shared_ptr<Node> max);
            void validate_and_infer_types() override;
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
            bool get_include_padding_in_avg_computation() const
            {
                return m_include_padding_in_avg_computation;
            }

        protected:
            Shape m_window_shape;
            Strides m_window_movement_strides;
            Shape m_padding_below;
            Shape m_padding_above;
            bool m_include_padding_in_avg_computation;
        };
    }
}
