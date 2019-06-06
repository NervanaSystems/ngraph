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

#include "ngraph/op/util/logical_reduction.hpp"

using namespace std;
using namespace ngraph;

op::util::LogicalReduction::LogicalReduction()
{
}

op::util::LogicalReduction::LogicalReduction(const Output<Node>& arg, const AxisSet& reduction_axes)
    : Op({arg})
{
    set_reduction_axes(reduction_axes);
}

op::util::LogicalReduction::LogicalReduction(const std::shared_ptr<Node>& arg,
                                             const AxisSet& reduction_axes)
    : Op(check_single_output_args({arg}))
{
    set_reduction_axes(reduction_axes);
}

op::util::LogicalReduction::LogicalReduction(const std::string& node_type,
                                             const std::shared_ptr<Node>& arg,
                                             const AxisSet& reduction_axes)
    : Op(node_type, check_single_output_args({arg}))
{
    set_reduction_axes(reduction_axes);
}

const AxisSet& op::util::LogicalReduction::get_reduction_axes() const
{
    return m_reduction_axes;
}

void op::util::LogicalReduction::set_reduction_axes(const AxisSet& reduction_axes)
{
    m_reduction_axes = reduction_axes;
}

void op::util::LogicalReduction::validate_and_infer_types()
{
    auto input_shape = get_input_partial_shape(0);
    auto input_rank = input_shape.rank();

    PartialShape result_shape{PartialShape::dynamic()};

    if (input_rank.is_static())
    {
        std::vector<Dimension> dims;

        for (auto axis : m_reduction_axes)
        {
            NODE_VALIDATION_CHECK(this,
                                  axis < size_t(input_rank),
                                  "Reduction axis (",
                                  axis,
                                  ") is out of bounds ",
                                  "(argument shape: ",
                                  input_shape,
                                  ", reduction axes: ",
                                  m_reduction_axes,
                                  ")");
        }

        for (size_t i = 0; i < size_t(input_rank); i++)
        {
            if (m_reduction_axes.count(i) == 0)
            {
                dims.push_back(input_shape[i]);
            }
        }

        result_shape = PartialShape(dims);
    }

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(0).compatible(element::boolean),
                          "Input element type must be boolean.");

    set_output_type(0, element::boolean, result_shape);
}
