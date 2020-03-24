//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Generic padding operation.
            class NGRAPH_API Pad : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"Pad", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a generic padding operation.
                Pad() = default;
                /// \brief Constructs a generic padding operation.
                ///
                /// \param arg The node producing input tensor to be padded.
                /// \param arg_pad_value The node producing the scalar value
                /// to be inserted for padding.
                /// \param padding_below The padding-below widths.
                /// \param padding_above The padding-above widths.
                /// \param pad_mode The padding mode: CONSTANT(default), EDGE, REFLECT or SYMMETRIC.
                Pad(const Output<Node>& arg,
                    const Output<Node>& arg_pad_value,
                    const CoordinateDiff& padding_below,
                    const CoordinateDiff& padding_above,
                    PadMode pad_mode = PadMode::CONSTANT);

                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;
                void validate_and_infer_types() override;
                /// \return The padding-below sizes.
                const CoordinateDiff& get_padding_below() const { return m_padding_below; }
                void set_padding_below(const CoordinateDiff& padding_below)
                {
                    m_padding_below = padding_below;
                }
                /// \return The padding-above sizes.
                const CoordinateDiff& get_padding_above() const { return m_padding_above; }
                void set_padding_above(const CoordinateDiff& padding_above)
                {
                    m_padding_above = padding_above;
                }

                /// \brief DEPRECATED. This is just a stub for backends that used to implement the
                ///        interior padding feature, which is no longer supported.
                /// \return Returns a shape full of zeros,
                /// with the same rank as get_padding_below().
                const Shape& get_padding_interior() const { return m_padding_interior_fake; }
                /// \return The padding mode.
                PadMode get_pad_mode() const { return m_pad_mode; }
                void set_pad_mode(PadMode pad_mode) { m_pad_mode = pad_mode; }
                /// \return The default value for Pad.
                virtual std::shared_ptr<Node> get_default_value() const override;

            protected:
                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;
                CoordinateDiff m_padding_below;
                CoordinateDiff m_padding_above;
                Shape m_padding_interior_fake; // LEGACY: This is all zeros.
                PadMode m_pad_mode;
            };
        }

        namespace v1
        {
            /// \brief Generic padding operation.
            class NGRAPH_API Pad : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"Pad", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a generic padding operation.
                ///
                /// \param arg The node producing input tensor to be padded.
                /// \param pads_begin The node which specifies the number of padding elements at the
                /// beginning of each axis
                /// \param pads_end The node which specifies the number of padding elements at the
                /// end of each axis
                /// \param arg_pad_value The node with value which set to extended elements
                /// if pad_mode is CONSTANT
                /// \param pad_mode The padding mode: CONSTANT, EDGE, REFLECT or SYMMETRIC.
                Pad(const Output<Node>& arg,
                    const Output<Node>& pads_begin,
                    const Output<Node>& pads_end,
                    const Output<Node>& arg_pad_value,
                    PadMode pad_mode);

                /// \brief Constructs a generic padding operation.
                ///
                /// \param arg The node producing input tensor to be padded.
                /// \param pads_begin The node which specifies the number of padding elements
                /// at the beginning of each axis
                /// \param pads_end The node which specifies the number of padding elements
                /// at the end of each axis
                /// \param pad_mode The padding mode: CONSTANT, EDGE, REFLECT or SYMMETRIC.
                Pad(const Output<Node>& arg,
                    const Output<Node>& pads_begin,
                    const Output<Node>& pads_end,
                    PadMode pad_mode);

                /// \brief Constructs a generic padding operation.
                Pad() = default;

                size_t get_version() const override { return 1; }
                void validate_and_infer_types() override;
                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;

                /// return The node which specifies the number of padding elements
                /// at the beginning of each axis
                CoordinateDiff get_pads_begin() const;
                /// return The node which specifies the number of padding elements
                /// at the end of each axis
                CoordinateDiff get_pads_end() const;

                /// \return The padding mode.
                PadMode get_pad_mode() const { return m_pad_mode; }
                void set_pad_mode(PadMode pad_mode) { m_pad_mode = pad_mode; }
            protected:
                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;

            private:
                PadMode m_pad_mode;
            };
        }

        // latest stable opset version
        using v0::Pad;
    }
}
