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
        class SoftmaxCrossEntropy : public ngraph::op::util::FusedOp
        {
        public:
            NGRAPH_API
            static constexpr NodeTypeInfo type_info{"SoftmaxCrossEntropy", 0};
            const NodeTypeInfo& get_type_info() const override { return type_info; }
            SoftmaxCrossEntropy() = default;
            /// \brief Softamx + CrossEntropy for numerical stabilization
            /// \param arg1 Node that produces the tensor to normalize
            /// \param arg2 Node that produces OneHot Lables
            /// \param reduction_axes axis on which to reduce the summation operation
            SoftmaxCrossEntropy(const Output<Node>& arg1,
                                const Output<Node>& arg2,
                                const AxisSet& reduction_axes);

            virtual NodeVector decompose_op() const override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            const AxisSet& get_summation_axis() const { return m_summation_axis; }
        private:
            AxisSet m_summation_axis;
        };
    } // namespace op
} // namespace ngraph
