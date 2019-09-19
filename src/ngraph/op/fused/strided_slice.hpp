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

#include "ngraph/node.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/op/util/fused_op.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief  Normalization input tensor with L2 norm.
        ///
        class StridedSlice : public ngraph::op::util::FusedOp
        {
        public:
            NGRAPH_API
            static constexpr NodeTypeInfo type_info{"StridedSlice", 0};
            const NodeTypeInfo& get_type_info() const override { return type_info; }
            StridedSlice() = default;
            // TODO DOC
            StridedSlice(const Output<Node>& data,
                         const Output<Node>& begin,
                         const Output<Node>& end,
                         const Output<Node>& stride,
                         const AxisSet& begin_mask,
                         const AxisSet& end_mask,
                         const AxisSet& new_axis_mask = AxisSet{},
                         const AxisSet& shrink_axis_mask = AxisSet{},
                         const AxisSet& ellipsis_mask = AxisSet{});

            // TODO DOC
            StridedSlice(const Output<Node>& data,
                         const Output<Node>& begin,
                         const Output<Node>& end,
                         const AxisSet& begin_mask,
                         const AxisSet& end_mask,
                         const AxisSet& new_axis_mask = AxisSet{},
                         const AxisSet& shrink_axis_mask = AxisSet{},
                         const AxisSet& ellipsis_mask = AxisSet{});

            NodeVector decompose_op() const override;
            void pre_validate_and_infer_types() override;

            std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

        private:
            AxisSet m_begin_mask;
            AxisSet m_end_mask;
            AxisSet m_new_axis_mask;
            AxisSet m_shrink_axis_mask;
            AxisSet m_ellipsis_mask;
        };
    }
}
