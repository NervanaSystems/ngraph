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
            /// \brief Softamax + CrossEntropy for numerical stabilization
            /// \param arg1 Node that produces the tensor to normalize
            /// \param arg2 Node that produces OneHot Lables
            /// \param reduction_axes axes on which to reduce the summation operation
            SoftmaxCrossEntropy(const Output<Node>& arg1,
                                const Output<Node>& arg2,
                                const AxisSet& reduction_axes);

            virtual NodeVector decompose_op() const override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            const AxisSet& get_reduction_axes() const { return m_reduction_axes; }
        private:
            AxisSet m_reduction_axes;
        };

        class SoftmaxCrossEntropyBackprop : public util::FusedOp
        {
        public:
            NGRAPH_API
            static constexpr NodeTypeInfo type_info{"SoftmaxCrossEntropyBackprop", 0};
            const NodeTypeInfo& get_type_info() const override { return type_info; }
            SoftmaxCrossEntropyBackprop() = default;

            /// \brief Backprop for SoftmaxCrossEntropy
            /// \param delta Node that produces the delta during bprop
            /// \param softmax Node that produces softmax from fprop
            /// \param onehot Node that produces OneHot Labels from fprop
            SoftmaxCrossEntropyBackprop(const Output<Node>& delta,
                                        const Output<Node>& softmax,
                                        const Output<Node>& onehot);

            virtual NodeVector decompose_op() const override;

            void pre_validate_and_infer_types() override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;
        };
    } // namespace op
} // namespace ngraph
