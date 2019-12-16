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
            /// \brief pdpd slice op
            ///
            class NGRAPH_API PartialSlice : public ngraph::op::util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"PartialSlice", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                PartialSlice() = default;
                /// \brief Constructs an PartialSlice operation.
                ///
                /// \param data Input tensor
                /// \param axes Axes that lower and upper bounds apply to
                /// \param lower_bounds Starting indices of corresponding axis in `axes`
                /// \param upper_bounds Ending indices of corresponding axis in `axes`
                /// \param decrease_axis Axes to be dropped (dimension will be one)
                PartialSlice(const Output<Node>& data,
                             const AxisVector& axes,
                             const std::vector<int64_t>& lower_bounds,
                             const std::vector<int64_t>& upper_bounds,
                             const AxisVector& decrease_axes);

                virtual NodeVector decompose_op() const override;

                void pre_validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;

                const AxisVector& get_axes() const { return m_axes; }
                const std::vector<int64_t>& get_lower_bounds() const { return m_lower_bounds; }
                const std::vector<int64_t>& get_upper_bounds() const { return m_upper_bounds; }
                const AxisVector& get_decrease_axes() const { return m_decrease_axes; }
            protected:
                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;

            private:
                AxisVector m_axes;
                std::vector<int64_t> m_lower_bounds;
                std::vector<int64_t> m_upper_bounds;
                AxisVector m_decrease_axes;
            };

            /// \brief pdpd slice backprop
            ///
            class NGRAPH_API PartialSliceBackprop : public ngraph::op::util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"PartialSliceBackprop", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                PartialSliceBackprop() = default;
                /// \brief Constructs an PartialSliceBackprop operation.
                ///
                /// \param data Input tensor
                /// \param dout Onput tensor from fprop
                /// \param axes Axes that lower and upper bounds apply to
                /// \param lower_bounds Starting indices of corresponding axis in `axes`
                /// \param upper_bounds Ending indices of corresponding axis in `axes`
                PartialSliceBackprop(const Output<Node>& data,
                                     const Output<Node>& dout,
                                     const AxisVector& axes,
                                     const std::vector<int64_t>& lower_bounds,
                                     const std::vector<int64_t>& upper_bounds);

                virtual NodeVector decompose_op() const override;

                void pre_validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;

                const AxisVector& get_axes() const { return m_axes; }
                const std::vector<int64_t>& get_lower_bounds() const { return m_lower_bounds; }
                const std::vector<int64_t>& get_upper_bounds() const { return m_upper_bounds; }
            private:
                AxisVector m_axes;
                std::vector<int64_t> m_lower_bounds;
                std::vector<int64_t> m_upper_bounds;
            };
        }
        using v0::PartialSlice;
        using v0::PartialSliceBackprop;
    }
}
