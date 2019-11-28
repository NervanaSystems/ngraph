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

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/fused_op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief LogSoftmax operation
            class NGRAPH_API LogSoftmax : public ngraph::op::util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"LogSoftmax", 0};
                LogSoftmax() = default;
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a LogSoftmax node.
                ///
                /// \param data Node that produces the first input tensor
                /// \param axis Describes the axis of the inputs when coerced to 2D
                LogSoftmax(const Output<Node>& data, int64_t axis);

                virtual NodeVector decompose_op() const override;

                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;

                int64_t get_axis() const { return m_axis; }
            protected:
                int64_t m_axis;
            };
        }
        using v0::LogSoftmax;
    } // namespace op
} // namespace ngraph
