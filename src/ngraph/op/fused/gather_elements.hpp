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
            class NGRAPH_API GatherElements : public ngraph::op::util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"GatherElements", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                GatherElements() = default;
                /// \brief CrossEntropy for computing loss
                /// \param arg1 Node that produces the input tensor
                /// \param arg2 Node that produces ground truth lables for the input
                /// \param soft_label flag indicating whether to interpretate the given labels as
                /// soft
                /// labels
                /// \param ignore_index Specifies a target value that is ignored and does not
                /// contribute
                /// to the input gradient Only valid if soft_label is set to False
                GatherElements(const Output<Node>& arg1,
                               const Output<Node>& arg2,
                               int64_t axis = 0);

                virtual NodeVector decompose_op() const override;

                void pre_validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;

                int64_t get_axis() const { return m_axis; }
            private:
                int64_t m_axis;
            };
        }
        using v0::GatherElements;
    } // namespace op
} // namespace ngraph