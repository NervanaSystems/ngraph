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

#include "ngraph/builder/split.hpp"
#include "ngraph/op/fused/split.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Split::type_info;

op::Split::Split(const Output<Node>& data, const int axis, const size_t num_split)
    : FusedOp({data})
    , m_split_evenly{true}
    , m_axis{axis}
    , m_num_split{num_split}
{
    constructor_validate_and_infer_types();
}

op::Split::Split(const Output<Node>& data, const int axis, const std::vector<size_t>& splits)
    : FusedOp({data})
    , m_split_evenly{false}
    , m_axis{axis}
    , m_num_split{0}
    , m_splits{splits}
{
    constructor_validate_and_infer_types();
}

void op::Split::pre_validate_and_infer_types()
{
    // Create dynamic-typed outputs. Actual shape/type will be computed during shape inference
    for (size_t i = 0; i < std::max(m_splits.size(), m_num_split); i++)
    {
        set_output_type(i, input(0).get_element_type(), PartialShape::dynamic());
    }

    if (is_dynamic())
    {
        return;
    }

    const auto shape = input(0).get_shape();

    m_axis = adjust_axis_value(m_axis, shape.size());
    NODE_VALIDATION_CHECK(this,
                          m_axis >= 0 && m_axis < static_cast<int64_t>(shape.size()),
                          "The 'axis' parameter for Split has to point to one of the "
                          "input tensor's shape dimensions.");

    const auto dimension_at_axis = shape.at(m_axis);
    if (m_split_evenly)
    {
        NODE_VALIDATION_CHECK(this,
                              dimension_at_axis % m_num_split == 0,
                              "The input tensor's dimension pointed by the 'axis' parameter: ",
                              dimension_at_axis,
                              " has to be a multiple of the 'num_split' parameter value: ",
                              m_num_split);

        m_splits.assign(m_num_split, dimension_at_axis / m_num_split);
    }
    else
    {
        const auto sum_splits = accumulate(begin(m_splits), end(m_splits), 0UL);
        NODE_VALIDATION_CHECK(this,
                              sum_splits == dimension_at_axis,
                              "The input tensor's dimension pointed by the 'axis' parameter: ",
                              dimension_at_axis,
                              " has to be equal to the sum of splits passed to the op: ",
                              sum_splits);

        const bool all_splits_positive =
            all_of(begin(m_splits), end(m_splits), [](const size_t v) { return v > 0; });

        NODE_VALIDATION_CHECK(this,
                              all_splits_positive == true,
                              "All values of the 'splits' attribute must be greater than zero");
    }
}

NodeVector op::Split::decompose_op() const
{
    return builder::split(input_value(0), m_splits, m_axis);
}

shared_ptr<Node> op::Split::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Split>(new_args.at(0), m_axis, m_splits);
}

size_t op::Split::adjust_axis_value(const int axis, const size_t input_tensor_rank) const
{
    if (axis < 0)
    {
        return axis + input_tensor_rank;
    }
    else
    {
        return axis;
    }
}
