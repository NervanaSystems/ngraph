//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include "ngraph/graph_util.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Batched average pooling operation, with optional padding and window stride.
        ///
        class AvgPool : public Op
        {
        public:
            NGRAPH_API
            static const std::string type_name;
            std::string description() const override { return type_name; }
            /// \brief Constructs a batched average pooling operation.
            AvgPool() = default;

            /// \brief Constructs a batched average pooling operation.
            ///
            /// \param arg The output producing the input data batch tensor.<br>
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
            /// \param pad_type Padding type to use for additional padded dimensions
            /// \param ceil_mode Whether to use ceiling while computing output shape.
            AvgPool(const Output<Node>& arg,
                    const Shape& window_shape,
                    const Strides& window_movement_strides,
                    const Shape& padding_below,
                    const Shape& padding_above,
                    bool include_padding_in_avg_computation,
                    const PadType& pad_type,
                    bool ceil_mode);

            /// \brief Constructs a batched average pooling operation.
            ///
            /// \param arg The output producing the input data batch tensor.<br>
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
            /// \param pad_type Padding type to use for additional padded dimensions
            AvgPool(const Output<Node>& arg,
                    const Shape& window_shape,
                    const Strides& window_movement_strides,
                    const Shape& padding_below,
                    const Shape& padding_above,
                    bool include_padding_in_avg_computation,
                    const PadType& pad_type);

            /// \brief Constructs a batched average pooling operation.
            ///
            /// \param arg The output producing the input data batch tensor.<br>
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
            AvgPool(const Output<Node>& arg,
                    const Shape& window_shape,
                    const Strides& window_movement_strides,
                    const Shape& padding_below,
                    const Shape& padding_above,
                    bool include_padding_in_avg_computation = false);

            /// \brief Constructs a batched, unpadded average pooling operation (i.e., all padding shapes are set to 0).
            ///
            /// \param arg The output producing the input data batch tensor.<br>
            /// `[d1, ..., dn]`
            /// \param window_shape The window shape.<br>
            /// `[n]`
            /// \param window_movement_strides The window movement strides.<br>
            /// `[n]`
            AvgPool(const Output<Node>& arg,
                    const Shape& window_shape,
                    const Strides& window_movement_strides);

            /// \brief Constructs an unstrided batched convolution operation (i.e., all window movement strides are 1 and all padding shapes are set to 0).
            ///
            /// \param arg The output producing the input data batch tensor.<br>
            /// `[d1, ..., dn]`
            /// \param window_shape The window shape.<br>
            /// `[n]`
            AvgPool(const Output<Node>& arg, const Shape& window_shape);

            void validate_and_infer_types() override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;

            /// \return The window shape.
            const Shape& get_window_shape() const;
            void set_window_shape(const Shape& window_shape);
            /// \return The window movement strides.
            const Strides& get_window_movement_strides() const;
            void set_window_movement_strides(const Strides& window_movement_strides);
            /// \return The below-padding shape.
            const Shape& get_padding_below() const;
            void set_padding_below(const Shape& padding_below);
            /// \return The above-padding shape.
            const Shape& get_padding_above() const;
            void set_padding_above(const Shape& padding_above);
            bool get_include_padding_in_avg_computation() const;
            void set_include_padding_in_avg_computation(bool include_padding_in_avg_computation);
            /// \return The pad type for pooling.
            const PadType& get_pad_type() const;
            void set_pad_type(const PadType& pad_type);
            bool get_ceil_mode() const;
            void set_ceil_mode(bool ceil_mode);
            /// \return The default value for AvgPool.
            virtual std::shared_ptr<Node> get_default_value() const override
            {
                return ngraph::make_constant_from_string("0", get_element_type(), get_shape());
            }

        protected:
            Shape m_window_shape;
            Strides m_window_movement_strides;
            Shape m_padding_below;
            Shape m_padding_above;
            bool m_include_padding_in_avg_computation{false};
            PadType m_pad_type{PadType::EXPLICIT};
            bool m_ceil_mode{false};
        };

        class AvgPoolBackprop : public Op
        {
        public:
            NGRAPH_API
            static const std::string type_name;
            std::string description() const override { return type_name; }
            AvgPoolBackprop() = default;
            AvgPoolBackprop(const Shape& forward_arg_shape,
                            const std::shared_ptr<Node>& delta,
                            const Shape& window_shape,
                            const Strides& window_movement_strides,
                            const Shape& padding_below,
                            const Shape& padding_above,
                            bool include_padding_in_avg_computation);

            void validate_and_infer_types() override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            const Shape& get_forward_arg_shape() const;
            void set_forward_arg_shape(const Shape& forward_arg_shape);
            const Shape& get_window_shape() const;
            void set_window_shape(const Shape& window_shape);
            const Strides& get_window_movement_strides() const;
            void set_window_movement_strides(const Strides& window_movement_strides);
            const Shape& get_padding_below() const;
            void set_padding_below(const Shape& padding_below);
            const Shape& get_padding_above() const;
            void set_padding_above(const Shape& padding_abve);
            bool get_include_padding_in_avg_computation() const;
            void set_include_padding_in_avg_computation(bool include_padding_in_avg_computation);

        protected:
            Shape m_forward_arg_shape;
            Shape m_window_shape;
            Strides m_window_movement_strides;
            Shape m_padding_below;
            Shape m_padding_above;
            bool m_include_padding_in_avg_computation{false};
        };
    }
}
