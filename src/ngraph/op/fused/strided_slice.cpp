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

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::StridedSlice::type_info;

op::StridedSlice::StridedSlice(const Output<Node>& data,
                               const Output<Node>& begin,
                               const Output<Node>& end,
                               const Output<Node>& stride,
                               std::vector<int64_t> begin_mask,
                               std::vector<int64_t> end_mask,
                               std::vector<int64_t> new_axis_mask,
                               std::vector<int64_t> shrink_axis_mask,
                               std::vector<int64_t> ellipsis_mask)
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
                               std::vector<int64_t> begin_mask,
                               std::vector<int64_t> end_mask,
                               std::vector<int64_t> new_axis_mask,
                               std::vector<int64_t> shrink_axis_mask,
                               std::vector<int64_t> ellipsis_mask)
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