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

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            /// \brief Elementwise logical negation operation.
            class NGRAPH_API LogicalNot : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"LogicalNot", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a logical negation operation.
                LogicalNot() = default;
                /// \brief Constructs a logical negation operation.
                ///
                /// \param arg Node that produces the input tensor.
                LogicalNot(const Output<Node>& arg);

                bool visit_attributes(AttributeVisitor& visitor) override;
                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;
            };
        }
        namespace v0
        {
            /// \brief Elementwise logical negation operation.
            class NGRAPH_API Not : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"Not", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a logical negation operation.
                Not() = default;
                /// \brief Constructs a logical negation operation.
                ///
                /// \param arg Node that produces the input tensor.
                Not(const Output<Node>& arg);

                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;
            };
        }

        using v0::Not;
    } // namespace op
} // namespace ngraph
