// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include "ngraph/ops/pad.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

#define SUBTRACT_OR_ZERO(x, y) (((y) > (x)) ? 0 : (x) - (y))

op::Pad::Pad(const std::shared_ptr<Node>& arg,
             const std::shared_ptr<Node>& arg_pad_value,
             const Shape& padding_below,
             const Shape& padding_above,
             const Shape& padding_interior)
    : RequiresTensorViewArgs("Pad", {arg, arg_pad_value})
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_padding_interior(padding_interior)
{
    if (get_input_element_type(0) != get_input_element_type(1))
    {
        throw ngraph_error("Pad argument tensor and padding value element types do not match");
    }

    if (get_input_shape(1) != Shape{})
    {
        throw ngraph_error("Padding value for pad is not a scalar");
    }

    auto arg_shape = get_input_shape(0);

    if (arg_shape.size() != padding_below.size())
    {
        throw ngraph_error("Pad rank for below-padding does not match rank of argument tensor");
    }

    if (arg_shape.size() != padding_above.size())
    {
        throw ngraph_error("Pad rank for above-padding does not match rank of argument tensor");
    }

    if (arg_shape.size() != padding_interior.size())
    {
        throw ngraph_error("Pad rank for interior padding does not match rank of argument tensor");
    }

    Shape result_shape;

    for (size_t i = 0; i < arg_shape.size(); i++)
    {
        result_shape.push_back(
            padding_below[i] +
            SUBTRACT_OR_ZERO(arg_shape[i] * (padding_interior[i] + 1), padding_interior[i]) +
            padding_above[i]);
    }

    set_value_type_checked(get_input_element_type(0), result_shape);
}

std::shared_ptr<Node>
    op::Pad::copy_with_new_args(const std::vector<std::shared_ptr<Node>>& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return std::make_shared<Pad>(
        new_args.at(0), new_args.at(1), m_padding_below, m_padding_above, m_padding_interior);
}

bool op::Pad::is_functionally_identical(const Node& other) const
{
    bool rc = true;
    if (Node::is_functionally_identical(other))
    {
        const Pad& rhs = dynamic_cast<const Pad&>(other);
        rc &= m_padding_below == rhs.m_padding_below;
        rc &= m_padding_above == rhs.m_padding_above;
        rc &= m_padding_interior == rhs.m_padding_interior;
    }
    else
    {
        rc = false;
    }
    return rc;
}

/* The "y" half of this is going to be a bit tricky... best way to handle it, I think, is to ReplaceSlice the non-padded values in the incoming delta tensor
   with a zero broadcasted to x's shape; then sum that and backprop the result to y.
void op::Pad::generate_adjoints(autodiff::Adjoints& adjoints, const std::shared_ptr<Node>& delta)
{
    auto x = get_inputs_op(0);
    auto y = get_inputs_op(1);

    
}
*/
