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

#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Batched max pooling operation, with optional padding and window stride.
        class MaxPool : public Op
        {
        public:
            NGRAPH_API
            static const std::string type_name;
            const std::string& description() const override { return type_name; }
            /// \brief Constructs a batched max pooling operation.
            MaxPool() = default;

            /// \brief Constructs a batched max pooling operation.
            ///
            /// \param arg The node producing the input data batch tensor.
            /// \param window_shape The window shape.
            /// \param window_movement_strides The window movement strides.
            /// \param padding_below The below-padding shape.
            /// \param padding_above The above-padding shape.
            /// \param pad_type The pad type for automatically computing padding sizes
            /// \param ceil_mode Whether to use ceiling while computing output shape.
            MaxPool(const Output<Node>& arg,
                    const Shape& window_shape,
                    const Strides& window_movement_strides,
                    const Shape& padding_below,
                    const Shape& padding_above,
                    const PadType& pad_type,
                    bool ceil_mode);

            /// \brief Constructs a batched max pooling operation.
            ///
            /// \param arg The node producing the input data batch tensor.
            /// \param window_shape The window shape.
            /// \param window_movement_strides The window movement strides.
            /// \param padding_below The below-padding shape.
            /// \param padding_above The above-padding shape.
            /// \param pad_type The pad type for automatically computing padding sizes
            MaxPool(const Output<Node>& arg,
                    const Shape& window_shape,
                    const Strides& window_movement_strides,
                    const Shape& padding_below,
                    const Shape& padding_above,
                    const PadType& pad_type);

            /// \brief Constructs a batched max pooling operation.
            ///
            /// \param arg The node producing the input data batch tensor.
            /// \param window_shape The window shape.
            /// \param window_movement_strides The window movement strides.
            /// \param padding_below The below-padding shape.
            /// \param padding_above The above-padding shape.
            MaxPool(const Output<Node>& arg,
                    const Shape& window_shape,
                    const Strides& window_movement_strides,
                    const Shape& padding_below,
                    const Shape& padding_above);

            void validate_and_infer_types() override;

            /// \brief Constructs a batched, unpadded max pooling operation (i.e., all padding
            ///        shapes are set to 0).
            ///
            /// \param arg The node producing the input data batch tensor.
            /// \param window_shape The window shape.
            /// \param window_movement_strides The window movement strides.
            MaxPool(const Output<Node>& arg,
                    const Shape& window_shape,
                    const Strides& window_movement_strides);

            /// \brief Constructs an unstrided batched max pooling operation (i.e., all window
            ///        movement strides are 1 and all padding shapes are set to 0).
            ///
            /// \param arg The node producing the input data batch tensor.
            /// \param window_shape The window shape.
            MaxPool(const Output<Node>& arg, const Shape& window_shape);

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            /// \return The window shape.
            const Shape& get_window_shape() const { return m_window_shape; }
            void set_window_shape(const Shape& window_shape) { m_window_shape = window_shape; }
            /// \return The window movement strides.
            const Strides& get_window_movement_strides() const { return m_window_movement_strides; }
            void set_window_movement_strides(const Strides& window_movement_strides)
            {
                m_window_movement_strides = window_movement_strides;
            }
            /// \return The below-padding shape.
            const Shape& get_padding_below() const { return m_padding_below; }
            void set_padding_below(const Shape& padding_below) { m_padding_below = padding_below; }
            /// \return The above-padding shape.
            const Shape& get_padding_above() const { return m_padding_above; }
            void set_adding_above(const Shape& padding_above) { m_padding_above = padding_above; }
            /// \return The pad type for pooling.
            const PadType& get_pad_type() const { return m_pad_type; }
            void set_pad_type(const PadType& pad_type) { m_pad_type = pad_type; }
            /// \return The ceiling mode being used for output shape computations
            bool get_ceil_mode() const { return m_ceil_mode; }
            void set_ceil_mode(bool ceil_mode) { m_ceil_mode = ceil_mode; }
            /// \return The default value for MaxPool.
            virtual std::shared_ptr<Node> get_default_value() const override;

        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;

            Shape m_window_shape;
            Strides m_window_movement_strides;
            Shape m_padding_below;
            Shape m_padding_above;
            PadType m_pad_type;
            bool m_ceil_mode{false};
        };

        class MaxPoolBackprop : public Op
        {
        public:
            NGRAPH_API
            static const std::string type_name;
            const std::string& description() const override { return type_name; }
            MaxPoolBackprop() = default;

            MaxPoolBackprop(const Output<Node>& arg_forward,
                            const Output<Node>& delta,
                            const Shape& window_shape,
                            const Strides& window_movement_strides,
                            const Shape& padding_below,
                            const Shape& padding_above);

            MaxPoolBackprop(const Output<Node>& arg_forward,
                            const Output<Node>& delta,
                            const Output<Node>& result_forward,
                            const Shape& window_shape,
                            const Strides& window_movement_strides,
                            const Shape& padding_below,
                            const Shape& padding_above);

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            void validate_and_infer_types() override;

            const Shape& get_window_shape() const { return m_window_shape; }
            void set_window_shape(const Shape& window_shape) { m_window_shape = window_shape; }
            const Strides& get_window_movement_strides() const { return m_window_movement_strides; }
            void set_window_movement_strides(const Strides& window_movement_strides)
            {
                m_window_movement_strides = window_movement_strides;
            }
            const Shape& get_padding_below() const { return m_padding_below; }
            void set_padding_below(const Shape& padding_below) { m_padding_below = padding_below; }
            const Shape& get_padding_above() const { return m_padding_above; }
            void set_padding_above(const Shape& padding_above) { m_padding_above = padding_above; }
        protected:
            Shape m_window_shape;
            Strides m_window_movement_strides;
            Shape m_padding_below;
            Shape m_padding_above;
        };
    }
}
