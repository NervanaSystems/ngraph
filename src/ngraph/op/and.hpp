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

#include <memory>

#include "ngraph/op/util/binary_elementwise_logical.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            /// \brief Elementwise logical-and operation.
            ///
            class LogicalAnd : public util::BinaryElementwiseLogical
            {
            public:
                NGRAPH_API
                static constexpr NodeTypeInfo type_info{"LogicalAnd", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a logical-and operation.
                LogicalAnd() = default;

                /// \brief Constructs a logical-and operation.
                ///
                /// \param arg0 Output that produces the first input tensor.<br>
                /// `[d0, ...]`
                /// \param arg1 Output that produces the second input tensor.<br>
                /// `[d0, ...]`
                /// \param auto_broadcast Auto broadcast specification
                ///
                /// Output `[d0, ...]`
                ///
                LogicalAnd(const Output<Node>& arg0,
                           const Output<Node>& arg1,
                           const AutoBroadcastSpec& auto_broadcast =
                               AutoBroadcastSpec(AutoBroadcastType::NUMPY));

                std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

                virtual bool is_commutative() const override { return true; }
            };
        } // namespace v0
        namespace v0
        {
            /// \brief Elementwise logical-and operation.
            ///
            class And : public util::BinaryElementwiseLogical
            {
            public:
                NGRAPH_API
                static constexpr NodeTypeInfo type_info{"And", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a logical-and operation.
                And() = default;

                /// \brief Constructs a logical-and operation.
                ///
                /// \param arg0 Output that produces the first input tensor.<br>
                /// `[d0, ...]`
                /// \param arg1 Output that produces the second input tensor.<br>
                /// `[d0, ...]`
                /// \param auto_broadcast Auto broadcast specification
                ///
                /// Output `[d0, ...]`
                ///
                And(const Output<Node>& arg0,
                    const Output<Node>& arg1,
                    const AutoBroadcastSpec& auto_broadcast = AutoBroadcastSpec());

                std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

                virtual bool is_commutative() const override { return true; }
            };
        }

        using v0::And;
    }
}
