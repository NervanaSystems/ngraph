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

    PartialShape inputs_shape_scheme{PartialShape::dynamic()};
    element::Type inputs_et{element::dynamic};
    Dimension concatenation_axis_output_dim{0};

    for (auto i = 0; i < get_inputs().size(); i++)
    {
        PartialShape this_input_shape = get_input_partial_shape(i);
        Dimension this_input_rank = this_input_shape.rank();
        if (this_input_rank.is_static())
        {
            NODE_VALIDATION_ASSERT(this, m_concatenation_axis < size_t(this_input_rank))
                << "Concatenation axis (" << m_concatenation_axis << ") is out of bounds for "
                << "argument " << i << ", which has shape " << this_input_shape << ".";

            concatenation_axis_output_dim += this_input_shape[m_concatenation_axis];
            this_input_shape[m_concatenation_axis] = Dimension::dynamic();

            NODE_VALIDATION_ASSERT(this,
                                   PartialShape::merge_into(inputs_shape_scheme, this_input_shape))
                << "Argument shapes are inconsistent; they must have the same rank, and must have "
                << "equal dimension everywhere except on the concatenation axis (axis "
                << m_concatenation_axis << ").";

            NODE_VALIDATION_ASSERT(
                this, element::Type::merge(inputs_et, inputs_et, get_input_element_type(i)))
                << "Argument element types are inconsistent.";
        }
        else
        {
            concatenation_axis_output_dim += Dimension::dynamic();
        }
    }

    PartialShape concatenated_shape = inputs_shape_scheme;

    if (concatenated_shape.rank().is_static())
    {
        concatenated_shape[m_concatenation_axis] = concatenation_axis_output_dim;
    }

    set_output_type(0, inputs_et, concatenated_shape);
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
