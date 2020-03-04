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

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Takes a slice of an input tensor, i.e., the sub-tensor that resides within a
            ///        bounding box, optionally with stride.
            class NGRAPH_API DynSlice : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"DynSlice", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                DynSlice() = default;
                /// \brief Constructs a dynamic tensor slice operation.
                ///
                /// \param arg The tensor to be sliced.
                /// \param lower_bounds The axiswise lower bounds of the slice (inclusive).
                /// \param upper_bounds The axiswise upper bounds of the slice (exclusive).
                /// \param strides The slicing strides; for example, strides of `{n,m}` means to
                /// take
                ///                every nth row and every mth column of the input matrix.
                /// \param lower_bounds_mask Ignores lower_bounds for axis with the mask set
                /// \param upper_bounds_mask Ignores upper_bounds for axis with the mask set
                /// \param new_axis          Add dimension one axis at the set positions
                /// \param shrink_axis       Delete dimensions at the set positions
                /// \param ellipsis_mask     Inserts missing dimensions on the set position
                DynSlice(const Output<Node>& arg,
                         const Output<Node>& lower_bounds,
                         const Output<Node>& upper_bounds,
                         const Output<Node>& strides,
                         const AxisSet& lower_bounds_mask = AxisSet{},
                         const AxisSet& upper_bounds_mask = AxisSet{},
                         const AxisSet& new_axis = AxisSet{},
                         const AxisSet& shrink_axis = AxisSet{},
                         const AxisSet& ellipsis_mask = AxisSet{});

                const AxisSet& get_lower_bounds_mask() const { return m_lower_bounds_mask; }
                const AxisSet& get_upper_bounds_mask() const { return m_upper_bounds_mask; }
                const AxisSet& get_new_axis() const { return m_new_axis; }
                const AxisSet& get_shrink_axis() const { return m_shrink_axis; }
                const AxisSet& get_ellipsis_mask() const { return m_ellipsis_mask; }
                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;
                void validate_and_infer_types() override;

            protected:
                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;

            private:
                /// Helper method to compute output shape
                Shape compute_output_shape() const;

                AxisSet m_lower_bounds_mask;
                AxisSet m_upper_bounds_mask;
                AxisSet m_new_axis;
                AxisSet m_shrink_axis;
                AxisSet m_ellipsis_mask;
            };
        }
        using v0::DynSlice;
    }
}
