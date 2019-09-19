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

#include "ngraph/op/fused/strided_slice.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/experimental/dyn_slice.hpp"

#include <algorithm>

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::StridedSlice::type_info;

op::StridedSlice::StridedSlice(const Output<Node>& data,
                               const Output<Node>& begin,
                               const Output<Node>& end,
                               const Output<Node>& stride,
                               const AxisSet& begin_mask,
                               const AxisSet& end_mask,
                               const AxisSet& new_axis_mask,
                               const AxisSet& shrink_axis_mask,
                               const AxisSet& ellipsis_mask)
    : FusedOp({data, begin, end, stride})
    , m_begin_mask{begin_mask}
    , m_end_mask{end_mask}
    , m_new_axis_mask{new_axis_mask}
    , m_shrink_axis_mask{shrink_axis_mask}
    , m_ellipsis_mask{ellipsis_mask}
{
    constructor_validate_and_infer_types();
}

op::StridedSlice::StridedSlice(const Output<Node>& data,
                               const Output<Node>& begin,
                               const Output<Node>& end,
                               const AxisSet& begin_mask,
                               const AxisSet& end_mask,
                               const AxisSet& new_axis_mask,
                               const AxisSet& shrink_axis_mask,
                               const AxisSet& ellipsis_mask)
    : FusedOp({data, begin, end})
    , m_begin_mask{begin_mask}
    , m_end_mask{end_mask}
    , m_new_axis_mask{new_axis_mask}
    , m_shrink_axis_mask{shrink_axis_mask}
    , m_ellipsis_mask{ellipsis_mask}
{
    constructor_validate_and_infer_types();
}

void op::StridedSlice::pre_validate_and_infer_types()
{
}

NodeVector op::StridedSlice::decompose_op() const
{
    Output<Node> strided_slice_result;
    const auto stride_value_provided = get_input_size() == 4;
    if (stride_value_provided)
    {
        strided_slice_result = make_shared<op::DynSlice>(input_value(0),
                                                         input_value(1),
                                                         input_value(2),
                                                         input_value(3),
                                                         m_begin_mask,
                                                         m_end_mask,
                                                         m_new_axis_mask,
                                                         m_shrink_axis_mask,
                                                         m_ellipsis_mask);
    }
    else
    {
        const auto added_axes =
            std::count_if(m_new_axis_mask.begin(), m_new_axis_mask.end(), [](size_t i) { return i == 1; });
        const auto shrinked_axes = std::count_if(
            m_shrink_axis_mask.begin(), m_shrink_axis_mask.end(), [](size_t i) { return i == 1; });
        const auto stride_size = m_begin_mask.size() + added_axes - shrinked_axes;
        ;
        const auto stride =
            op::Constant::create(element::i64, Shape{stride_size}, vector<int64_t>(stride_size, 1));
        strided_slice_result = make_shared<op::DynSlice>(input_value(0),
                                                         input_value(1),
                                                         input_value(2),
                                                         stride,
                                                         m_begin_mask,
                                                         m_end_mask,
                                                         m_new_axis_mask,
                                                         m_shrink_axis_mask,
                                                         m_ellipsis_mask);
    }

    return as_node_vector({strided_slice_result});
}

shared_ptr<Node> op::StridedSlice::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() == 3)
    {
        return make_shared<StridedSlice>(new_args.at(0),
                                         new_args.at(1),
                                         new_args.at(2),
                                         m_begin_mask,
                                         m_end_mask,
                                         m_new_axis_mask,
                                         m_shrink_axis_mask,
                                         m_ellipsis_mask);
    }
    if (new_args.size() == 4)
    {
        return make_shared<StridedSlice>(new_args.at(0),
                                         new_args.at(1),
                                         new_args.at(2),
                                         new_args.at(3),
                                         m_begin_mask,
                                         m_end_mask,
                                         m_new_axis_mask,
                                         m_shrink_axis_mask,
                                         m_ellipsis_mask);
    }
    throw ngraph_error("Incorrect number of new arguments");
}