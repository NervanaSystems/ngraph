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

#include <memory>

#include "ngraph/op/concat.hpp"
#include "ngraph/op/slice.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Concat::type_info;

op::Concat::Concat(const OutputVector& args, int64_t axis)
    : Op(args)
    , m_axis(axis)
{
    constructor_validate_and_infer_types();
}

op::Concat::Concat(const NodeVector& args, int64_t axis)
    : Concat(as_output_vector(args), axis)
{
}

bool op::Concat::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("axis", m_axis);
    return true;
}

void op::Concat::validate_and_infer_types()
{
    NODE_VALIDATION_CHECK(this, get_input_size() >= 1, "At least one argument required.");

    PartialShape inputs_shape_scheme{PartialShape::dynamic()};
    element::Type inputs_et{element::dynamic};
    Dimension concatenation_axis_output_dim{0};

    for (uint64_t i = 0; i < get_input_size(); i++)
    {
        NODE_VALIDATION_CHECK(this,
                              element::Type::merge(inputs_et, inputs_et, get_input_element_type(i)),
                              "Argument element types are inconsistent.");
        PartialShape this_input_shape = get_input_partial_shape(i);
        Dimension this_input_rank = this_input_shape.rank();
        if (this_input_rank.is_static())
        {
            if (get_concatenation_axis() < 0)
            {
                set_concatenation_axis(get_axis() < 0
                                           ? get_axis() + static_cast<int64_t>(this_input_rank)
                                           : get_axis());
            }
            auto concat_axis = get_concatenation_axis();
            NODE_VALIDATION_CHECK(this,
                                  concat_axis < static_cast<int64_t>(this_input_rank),
                                  "Concatenation axis (",
                                  concat_axis,
                                  ") is out of bounds for ",
                                  "argument ",
                                  i,
                                  ", which has shape ",
                                  this_input_shape,
                                  ".");

            concatenation_axis_output_dim += this_input_shape[concat_axis];
            this_input_shape[concat_axis] = Dimension::dynamic();

            NODE_VALIDATION_CHECK(
                this,
                PartialShape::merge_into(inputs_shape_scheme, this_input_shape),
                "Argument shapes are inconsistent; they must have the same rank, and must have ",
                "equal dimension everywhere except on the concatenation axis (axis ",
                concat_axis,
                ").");
        }
        else
        {
            concatenation_axis_output_dim += Dimension::dynamic();
        }
    }
    PartialShape concatenated_shape = inputs_shape_scheme;

    if (concatenated_shape.rank().is_static())
    {
        concatenated_shape[get_concatenation_axis()] = concatenation_axis_output_dim;
        set_output_type(0, inputs_et, concatenated_shape);
    }
    else
    {
        set_output_type(0, inputs_et, PartialShape::dynamic(concatenation_axis_output_dim));
    }
}

shared_ptr<Node> op::Concat::copy_with_new_args(const NodeVector& new_args) const
{
    // TODO(amprocte): Should we check the new_args count here?
    return make_shared<Concat>(new_args, m_axis);
}

void op::Concat::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto concat_result_shape = output(0).get_shape();

    Coordinate arg_delta_slice_lower = Coordinate(concat_result_shape.size(), 0);
    Coordinate arg_delta_slice_upper = concat_result_shape;
    Coordinate arg_delta_slice_strides = Coordinate(concat_result_shape.size(), 1);

    size_t pos = 0;

    for (auto value : input_values())
    {
        auto arg_shape = value.get_shape();

        auto slice_width = arg_shape[m_axis];

        size_t next_pos = pos + slice_width;
        arg_delta_slice_lower[m_axis] = pos;
        arg_delta_slice_upper[m_axis] = next_pos;

        adjoints.add_delta(
            value,
            make_shared<op::Slice>(
                delta, arg_delta_slice_lower, arg_delta_slice_upper, arg_delta_slice_strides));

        pos = next_pos;
    }
}
