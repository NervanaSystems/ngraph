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

#include "ngraph/op/util/binary_elementwise_comparison.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Elementwise greater-than operation.
            class NGRAPH_API Greater : public util::BinaryElementwiseComparison
            {
            public:
                static constexpr NodeTypeInfo type_info{"Greater", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a greater-than operation.
                Greater() = default;
                /// \brief Constructs a greater-than operation.
                ///
                /// \param arg0 Node that produces the first input tensor.
                /// \param arg1 Node that produces the second input tensor.
                /// \param auto_broadcast Auto broadcast specification
                Greater(const Output<Node>& arg0,
                        const Output<Node>& arg1,
                        const AutoBroadcastSpec& auto_broadcast = AutoBroadcastSpec());

                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;
            };
        } // namespace v0

        namespace v1
        {
            /// \brief Elementwise greater-than operation.
            class NGRAPH_API Greater : public util::BinaryElementwiseComparison
            {
            public:
                static constexpr NodeTypeInfo type_info{"Greater", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a greater-than operation.
                Greater() = default;
                /// \brief Constructs a greater-than operation.
                ///
                /// \param arg0 Node that produces the first input tensor.
                /// \param arg1 Node that produces the second input tensor.
                /// \param auto_broadcast Auto broadcast specification
                Greater(const Output<Node>& arg0,
                        const Output<Node>& arg1,
                        const AutoBroadcastSpec& auto_broadcast =
                            AutoBroadcastSpec(AutoBroadcastType::NUMPY));

                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;
                size_t get_version() const override { return 1; }
            };
        } // namespace v1

        using v0::Greater;
    }
}
