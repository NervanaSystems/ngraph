//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include <cassert>
#include <memory>

#include "ngraph/op/concat.hpp"
#include "ngraph/op/slice.hpp"

using namespace std;
using namespace ngraph;

op::Concat::Concat(const NodeVector& args, size_t concatenation_axis)
    : Op("Concat", check_single_output_args(args))
    , m_concatenation_axis(concatenation_axis)
{
    constructor_validate_and_infer_types();
}

void op::Concat::validate_and_infer_types()
{
    NODE_VALIDATION_ASSERT(this, m_inputs.size() >= 1) << "At least one argument required.";

    Shape first_input_shape = get_input_shape(0);
    size_t expected_rank = first_input_shape.size();
    element::Type expected_et = get_input_element_type(0);

    for (auto i = 1; i < get_inputs().size(); i++)
    {
        NODE_VALIDATION_ASSERT(this, get_input_shape(i).size() == expected_rank)
            << "Not all arguments have the same rank: argument 0 has shape " << first_input_shape
            << " of rank " << expected_rank << " but argument " << i << " has shape "
            << get_input_shape(i) << " of rank " << get_input_shape(i).size() << ".";

        NODE_VALIDATION_ASSERT(this, get_input_element_type(i) == expected_et)
            << "Not all arguments have the same element type: argument 0 has element type "
            << expected_et << " but argument " << i << " has element type "
            << get_input_element_type(i) << ".";
    }

    NODE_VALIDATION_ASSERT(this, m_concatenation_axis < expected_rank)
        << "Concatenation axis (" << m_concatenation_axis << ") is out of bounds (inputs have rank "
        << expected_rank << ").";

    size_t concatenation_axis_output_length = first_input_shape.at(m_concatenation_axis);

    for (auto i = 1; i < get_inputs().size(); i++)
    {
        for (auto j = 0; j < get_input_shape(i).size(); j++)
        {
            if (j != m_concatenation_axis)
            {
                NODE_VALIDATION_ASSERT(this, first_input_shape[j] == get_input_shape(i)[j])
                    << "Dimensions of argument " << i << " do not match for axis " << j
                    << " (expected " << first_input_shape[j] << ", got " << get_input_shape(i)[j]
                    << ").";
            }
            else
            {
                concatenation_axis_output_length += get_input_shape(i)[j];
            }
        }
    }

    Shape concatenated_shape = first_input_shape;
    concatenated_shape[m_concatenation_axis] = concatenation_axis_output_length;

    set_output_type(0, expected_et, concatenated_shape);
}

shared_ptr<Node> op::Concat::copy_with_new_args(const NodeVector& new_args) const
{
    // TODO(amprocte): Should we check the new_args count here?
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

    for (auto arg : get_arguments())
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
