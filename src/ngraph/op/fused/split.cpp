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
#include <numeric>

#include "ngraph/op/fused/split.hpp"
#include "ngraph/op/util/reshape.hpp"

using namespace std;
using namespace ngraph;

op::Split::Split(const shared_ptr<Node>& data, const int axis, const size_t num_split)
    : FusedOp("Split", {data})
    , m_split_evenly{true}
    , m_num_split{num_split}
{
    m_axis = adjust_axis_value(axis, data->get_shape().size());

    constructor_validate_and_infer_types();

    m_splits.assign(m_num_split, data->get_shape().at(m_axis) / m_num_split);
}

op::Split::Split(const std::shared_ptr<ngraph::Node>& data, const int axis, const std::vector<size_t>& splits)
        : FusedOp("Split", {data})
        , m_split_evenly{false}
        , m_splits{splits}
{
    m_axis = adjust_axis_value(axis, data->get_shape().size());

    constructor_validate_and_infer_types();
}

void op::Split::pre_validate_and_infer_types()
{
    const auto shape = get_argument(0)->get_shape();
    const auto dimension_at_axis = shape.at(m_axis);

    NODE_VALIDATION_CHECK(this,
                          m_axis >= 0 && m_axis < shape.size(),
                          "The 'axis' parameter for Split has to point to one of the "
                          "input tensor's shape dimensions.");

    if (m_split_evenly)
    {
        NODE_VALIDATION_CHECK(
                this,
                dimension_at_axis % m_num_split == 0,
                "The input tensor's dimension pointed by the 'axis' parameter: ",
                dimension_at_axis,
                " has to be a multiple of the 'num_split' parameter value: ",
                m_num_split);
    }
    else
    {
        const auto sum_splits = accumulate(begin(m_splits), end(m_splits), 0UL);
        NODE_VALIDATION_CHECK(
                this,
                sum_splits == dimension_at_axis,
                "The input tensor's dimension pointed by the 'axis' parameter: ",
                dimension_at_axis,
                " has to be equal to the sum of splits passed to the op: ",
                sum_splits);
    }
}

NodeVector op::Split::decompose_op() const
{
    return op::util::split(get_argument(0), m_splits, m_axis);
}

shared_ptr<Node> op::Split::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Expected 1 element in new_args for the Split op but got " + std::to_string(new_args.size()));
    }

    return make_shared<Split>(new_args.at(0), m_axis, m_num_split);
}

size_t op::Split::adjust_axis_value(const int axis, const size_t input_tensor_rank) const {
    if (axis < 0)
    {
        return axis + input_tensor_rank;
    }
    else
    {
        return axis;
    }
}
