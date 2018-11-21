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

#include "ngraph/op/pad.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

op::Pad::Pad(const shared_ptr<Node>& arg,
             const shared_ptr<Node>& arg_pad_value,
             const Shape& padding_below,
             const Shape& padding_above,
             const Shape& padding_interior)
    : Op("Pad", check_single_output_args({arg, arg_pad_value}))
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_padding_interior(padding_interior)
{
    constructor_validate_and_infer_types();
}

void op::Pad::validate_and_infer_types()
{
    element::Type result_et;

    NODE_VALIDATION_ASSERT(
        this, element::Type::merge(result_et, get_input_element_type(0), get_input_element_type(1)))
        << "Argument element types do not match (arg0 element type: " << get_input_element_type(0)
        << ", arg1 element type: " << get_input_element_type(1) << ").";

    NODE_VALIDATION_ASSERT(this, get_input_partial_shape(1).compatible(PartialShape{}))
        << "Argument for padding value is not a scalar (shape: " << get_input_partial_shape(1)
        << ").";

    auto arg_shape = get_input_partial_shape(0);

    NODE_VALIDATION_ASSERT(this,
                           m_padding_below.size() == m_padding_above.size() &&
                               m_padding_below.size() == m_padding_interior.size())
        << "Ranks for padding below (" << m_padding_below << "), padding above (" << m_padding_above
        << ") and interior padding (" << m_padding_interior << ") "
        << "do not match.";

    size_t implied_rank = m_padding_below.size();

    NODE_VALIDATION_ASSERT(this, arg_shape.rank().compatible(implied_rank))
        << "Rank for padding below/padding above/interior padding does not match the rank of the "
        << "data argument (padding below: " << m_padding_below << ", "
        << ", padding above: " << m_padding_above << ", interior padding: " << m_padding_interior
        << ").";

    std::vector<Dimension> result_dims(implied_rank, Dimension::dynamic());

    if (arg_shape.rank().is_static())
    {
        for (size_t i = 0; i < implied_rank; i++)
        {
            if (arg_shape[i].is_static())
            {
                result_dims[i] =
                    m_padding_below[i] +
                    subtract_or_zero(size_t(arg_shape[i]) * (m_padding_interior[i] + 1),
                                     m_padding_interior[i]) +
                    m_padding_above[i];
            }
        }
    }

    set_output_type(0, result_et, PartialShape(result_dims));
}

shared_ptr<Node> op::Pad::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Pad>(
        new_args.at(0), new_args.at(1), m_padding_below, m_padding_above, m_padding_interior);
}

/* The "y" half of this is going to be a bit tricky... best way to handle it, I think,
   is to ReplaceSlice the non-padded values in the incoming delta tensor with a zero
   broadcasted to x's shape; then sum that and backprop the result to y.

   For example, let's say we are padding a 2x2 with 1 above, below, and interior, and
   the deltas coming back are:

   d00 d01 d02 d03 d04
   d10 d11 d12 d13 d14
   d20 d21 d22 d23 d24
   d30 d31 d32 d33 d34
   d40 d41 d42 d43 d44

   We know that everything but d11, d13, d31, and d33 on the forward prop is just "y".
   So we mask that off (using the forward-prop padding values to determine start, end,
   and slice stride):

   d00 d01 d02 d03 d04
   d10   0 d12   0 d14
   d20 d21 d22 d23 d24
   d30   0 d32   0 d34
   d40 d41 d42 d43 d44

   Then sum that up:

   d00 + d01 + d02 + d03 + d04 +
   d10 +   0 + d12 +   0 + d14 +
   d20 + d21 + d22 + d23 + d24 +
   d30 +   0 + d32 +   0 + d34 +
   d40 + d41 + d42 + d43 + d44

   For the "x" backprop it's sort of the opposite; just slice out:

   d11 d13
   d31 d33

   and push that back.
*/
void op::Pad::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    throw invalid_argument("Autodiff is not yet implemented for Pad");
}

std::shared_ptr<Node> op::Pad::get_default_value() const
{
    AxisSet axes{};
    for (size_t i = 0; i < get_shape().size(); i++)
    {
        axes.insert(i);
    }
    return std::make_shared<op::Broadcast>(
        m_inputs.at(1).get_output().get_node(), get_shape(), axes);
}
