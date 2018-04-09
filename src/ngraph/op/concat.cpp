/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <cassert>
#include <memory>

#include "ngraph/op/concat.hpp"
#include "ngraph/op/slice.hpp"

using namespace std;
using namespace ngraph;

op::Concat::Concat(const NodeVector& args, size_t concatenation_axis)
    : RequiresTensorViewArgs("Concat", args)
    , m_concatenation_axis(concatenation_axis)
{
    if (m_inputs.size() < 1)
    {
        throw ngraph_error("At least one argument required");
    }

    auto& input_0 = get_inputs().at(0);
    auto input_0_shape = input_0.get_shape();
    if (m_concatenation_axis >= input_0_shape.size())
    {
        throw ngraph_error("Concatenation axis is out of bounds");
    }

    size_t concatenation_axis_length = input_0_shape.at(m_concatenation_axis);
    auto& input_0_element_type = input_0.get_element_type();

    for (auto i = 1; i < get_inputs().size(); i++)
    {
        auto& input_i = get_inputs().at(i);
        auto input_i_shape = input_i.get_shape();
        if (input_i_shape.size() != input_0_shape.size())
        {
            throw ngraph_error("Arguments to concat do not have same rank");
        }

        if (input_i.get_element_type() != input_0_element_type)
        {
            throw ngraph_error("Argument element types do not match");
        }

        for (auto j = 0; j < input_i_shape.size(); j++)
        {
            if (j != m_concatenation_axis && input_0_shape.at(j) != input_i_shape.at(j))
            {
                throw ngraph_error(
                    "Arguments to concat do not have same dimension on a non-concatenation axis");
            }
            else if (j == m_concatenation_axis)
            {
                concatenation_axis_length += input_i_shape.at(j);
            }
        }
    }
    vector<size_t> concatenated_shape = input_0_shape;
    concatenated_shape.at(m_concatenation_axis) = concatenation_axis_length;

    set_value_type_checked(make_shared<TensorViewType>(input_0_element_type, concatenated_shape));
}

shared_ptr<Node> op::Concat::copy_with_new_args(const NodeVector& new_args) const
{
    return make_shared<Concat>(new_args, m_concatenation_axis);
}

void op::Concat::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto concat_result_shape = get_outputs().at(0).get_shape();

    Coordinate arg_delta_slice_lower = Coordinate(concat_result_shape.size(), 0);
    Coordinate arg_delta_slice_upper = concat_result_shape;
    Coordinate arg_delta_slice_strides = Coordinate(concat_result_shape.size(), 1);

    size_t pos = 0;

    for (auto arg : get_input_ops())
    {
        auto arg_shape = arg->get_shape();

        auto slice_width = arg_shape[m_concatenation_axis];

        size_t next_pos = pos + slice_width;

        arg_delta_slice_lower[m_concatenation_axis] = pos;
        arg_delta_slice_upper[m_concatenation_axis] = next_pos;

        adjoints.add_delta(
            arg,
            make_shared<op::Slice>(
                delta, arg_delta_slice_lower, arg_delta_slice_upper, arg_delta_slice_strides));

        pos = next_pos;
    }
}
