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
#include <vector>

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            /// \brief Takes a slice of an input tensor, i.e., the sub-tensor that resides within a
            ///        bounding box, optionally with stride.
            class NGRAPH_API StridedSlice : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"Slice", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                StridedSlice() = default;

                /// \brief Constructs a dynamic tensor strided slice operation.
                ///
                /// \param data             The tensor to be sliced.
                /// \param begin            1D tensor with begin indexes for input blob slicing.
                /// \param end              1D tensor with end indexes for input blob slicing.
                /// \param strides          The slicing strides; for example, strides of `{n,m}`
                ///                         means to take every nth row and every mth column
                ///                         of the input matrix.
                /// \param begin_mask       When begin_mask[i] equal to 1 means that the
                ///                         corresponding dimension of the begin input is ignored.
                /// \param end_mask         When end_mask[i] is 1, the corresponding dimension of
                ///                         the end input is ignored.
                /// \param new_axis_mask    If new_axis_mask[i] is 1, a length 1 dimension
                ///                         is inserted on the i-th position.
                /// \param shrink_axis_mask If shrink_axis_mask[i] is 1, the dimension
                ///                         on the i-th position is deleted.
                /// \param ellipsis_mask    It inserts missing dimensions
                ///                         on a position of a non-zero bit.
                StridedSlice(const Output<Node>& data,
                             const Output<Node>& begin,
                             const Output<Node>& end,
                             const Output<Node>& strides,
                             const std::vector<int64_t>& begin_mask,
                             const std::vector<int64_t>& end_mask,
                             const std::vector<int64_t>& new_axis_mask = std::vector<int64_t>{},
                             const std::vector<int64_t>& shrink_axis_mask = std::vector<int64_t>{},
                             const std::vector<int64_t>& ellipsis_mask = std::vector<int64_t>{});

                /// \brief Constructs a dynamic tensor strided slice operation.
                ///
                /// \param data             The tensor to be sliced.
                /// \param begin            1D tensor with begin indexes for input blob slicing.
                /// \param end              1D tensor with end indexes for input blob slicing.
                /// \param begin_mask       When begin_mask[i] equal to 1 means that the
                ///                         corresponding dimension of the begin input is ignored.
                /// \param end_mask         When end_mask[i] is 1, the corresponding dimension of
                ///                         the end input is ignored.
                /// \param new_axis_mask    If new_axis_mask[i] is 1, a length 1 dimension
                ///                         is inserted on the i-th position.
                /// \param shrink_axis_mask If shrink_axis_mask[i] is 1, the dimension
                ///                         on the i-th position is deleted.
                /// \param ellipsis_mask    It inserts missing dimensions
                ///                         on a position of a non-zero bit.
                StridedSlice(const Output<Node>& data,
                             const Output<Node>& begin,
                             const Output<Node>& end,
                             const std::vector<int64_t>& begin_mask,
                             const std::vector<int64_t>& end_mask,
                             const std::vector<int64_t>& new_axis_mask = std::vector<int64_t>{},
                             const std::vector<int64_t>& shrink_axis_mask = std::vector<int64_t>{},
                             const std::vector<int64_t>& ellipsis_mask = std::vector<int64_t>{});

                const std::vector<int64_t>& get_begin_mask() const { return m_begin_mask; }
                const std::vector<int64_t>& get_end_mask() const { return m_end_mask; }
                const std::vector<int64_t>& get_new_axis_mask() const { return m_new_axis_mask; }
                const std::vector<int64_t>& get_shrink_axis_mask() const
                {
                    return m_shrink_axis_mask;
                }
                const std::vector<int64_t>& get_ellipsis_mask() const { return m_ellipsis_mask; }
                std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;
                void validate_and_infer_types() override;
                size_t get_version() const override { return 1; }
            protected:
                void generate_adjoints(autodiff::Adjoints& adjoints,
                                       const OutputVector& deltas) override;

            private:
                AxisSet convert_mask_to_axis_set(const std::vector<int64_t>& mask) const;

                std::vector<int64_t> m_begin_mask;
                std::vector<int64_t> m_end_mask;
                std::vector<int64_t> m_new_axis_mask;
                std::vector<int64_t> m_shrink_axis_mask;
                std::vector<int64_t> m_ellipsis_mask;
            };
        }
    }
}
