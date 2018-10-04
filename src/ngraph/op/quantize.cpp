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

#include "ngraph/op/quantize.hpp"

using namespace std;
using namespace ngraph;

op::Quantize::Quantize(shared_ptr<Node> input,
                       shared_ptr<Node> scale,
                       shared_ptr<Node> offset,
                       const element::Type& type,
                       const AxisSet& axes,
                       RoundMode round_mode)

    : Op("Quantize", check_single_output_args({input, scale, offset}))
    , m_type(type)
    , m_axes(axes)
    , m_round_mode(round_mode)
{
    constructor_validate_and_infer_types();
}

void op::Quantize::validate_and_infer_types()
{
    enum
    {
        INPUT,
        SCALE,
        OFFSET
    };

    set_output_size(1);
    set_output_type(0, m_type, get_input_shape(INPUT));

    NODE_VALIDATION_ASSERT(this, m_type.is_quantized()) << "Output element type (" << m_type
                                                        << ") must be a quantized type";

    NODE_VALIDATION_ASSERT(this, get_input_element_type(INPUT).is_real())
        << "Input element type (" << get_input_element_type(INPUT)
        << ") must be a floating point number";

    NODE_VALIDATION_ASSERT(this, get_input_element_type(SCALE).is_real())
        << "Scale element type (" << get_input_element_type(SCALE)
        << ") must be a floating point number";

    NODE_VALIDATION_ASSERT(this, get_input_element_type(OFFSET) == m_type)
        << "Offset element type (" << get_input_element_type(OFFSET)
        << ") must match output element type (" << m_type << ")";

    for (auto axis : m_axes)
    {
        NODE_VALIDATION_ASSERT(this, axis < get_shape().size())
            << "Quantizaztion axis (" << axis << ") is greater than input shape rank ("
            << get_shape().size() << ")";
    }

    Shape projected_shape = project(get_input_shape(INPUT), m_axes);

    NODE_VALIDATION_ASSERT(this, get_input_shape(SCALE) == projected_shape)
        << "Scale shape (" << get_input_shape(SCALE)
        << ") must match input shape projected along the quantization axes (" << projected_shape
        << ")";

    NODE_VALIDATION_ASSERT(this, get_input_shape(OFFSET) == projected_shape)
        << "Offset shape (" << get_input_shape(OFFSET)
        << ") must match input shape projected along the quantization axes (" << projected_shape
        << ")";

    NODE_VALIDATION_ASSERT(this, m_round_mode == RoundMode::HALF_AWAY_FROM_ZERO)
        << "Only RoundMode = HALF_AWAY_FROM_ZERO is supported, for now";
}

shared_ptr<Node> op::Quantize::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Quantize>(
        new_args.at(0), new_args.at(1), new_args.at(2), m_type, m_axes, m_round_mode);
}

void op::Quantize::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    throw ngraph_error("Forward-propagation-only operation");
}
