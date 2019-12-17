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

#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Elementwise division operation.
            class NGRAPH_API Divide : public util::BinaryElementwiseArithmetic
            {
            public:
                static constexpr NodeTypeInfo type_info{"Divide", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a division operation.
                Divide()
                    : util::BinaryElementwiseArithmetic(AutoBroadcastSpec::NONE)
                {
                }
                /// \brief Constructs a division operation.
                ///
                /// \param arg0 Node that produces the first input tensor.
                /// \param arg1 Node that produces the second input tensor.
                /// \param pythondiv Use Python style rounding for integral type
                /// \param auto_broadcast Auto broadcast specification
                Divide(const Output<Node>& arg0,
                       const Output<Node>& arg1,
                       bool pythondiv,
                       const AutoBroadcastSpec& auto_broadcast = AutoBroadcastSpec());

                /// \brief Constructs a division operation.
                ///
                /// \param arg0 Node that produces the first input tensor.
                /// \param arg1 Node that produces the second input tensor.
                /// \param auto_broadcast Auto broadcast specification
                Divide(const Output<Node>& arg0,
                       const Output<Node>& arg1,
                       const AutoBroadcastSpec& auto_broadcast = AutoBroadcastSpec());

                bool is_pythondiv() const { return m_pythondiv; }
                void set_is_pythondiv(bool pythondiv) { m_pythondiv = pythondiv; }
                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;

                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;

            protected:
                bool m_pythondiv{true};
            };
        } // namespace v0

        namespace v1
        {
            /// \brief Elementwise division operation.
            class NGRAPH_API Divide : public util::BinaryElementwiseArithmetic
            {
            public:
                static constexpr NodeTypeInfo type_info{"Divide", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a division operation.
                Divide()
                    : util::BinaryElementwiseArithmetic(AutoBroadcastSpec::NUMPY)
                {
                }

                /// \brief Constructs a division operation.
                ///
                /// \param arg0 Node that produces the first input tensor.
                /// \param arg1 Node that produces the second input tensor.
                /// \param pythondiv Use Python style rounding for integral type
                /// \param auto_broadcast Auto broadcast specification
                Divide(const Output<Node>& arg0,
                       const Output<Node>& arg1,
                       bool pythondiv,
                       const AutoBroadcastSpec& auto_broadcast =
                           AutoBroadcastSpec(AutoBroadcastType::NUMPY));

                /// \brief Constructs a division operation.
                ///
                /// \param arg0 Node that produces the first input tensor.
                /// \param arg1 Node that produces the second input tensor.
                /// \param auto_broadcast Auto broadcast specification
                Divide(const Output<Node>& arg0,
                       const Output<Node>& arg1,
                       const AutoBroadcastSpec& auto_broadcast =
                           AutoBroadcastSpec(AutoBroadcastType::NUMPY));

                bool is_pythondiv() const { return m_pythondiv; }
                void set_is_pythondiv(bool pythondiv) { m_pythondiv = pythondiv; }
                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;

                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;
                size_t get_version() const override { return 1; }
            protected:
                bool m_pythondiv{true};
            };
        } // namespace v1

        using v0::Divide;
    } // namespace op

    std::shared_ptr<Node> operator/(const Output<Node>& arg0, const Output<Node>& arg1);
} // namespace ngraph
