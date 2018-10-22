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

#include "ngraph/op/dequantize.hpp"
#include "ngraph/shape_util.hpp"

using namespace std;
using namespace ngraph;

op::Dequantize::Dequantize(shared_ptr<Node> input,
                           shared_ptr<Node> scale,
                           shared_ptr<Node> offset,
                           const element::Type& type,
                           const AxisSet& axes)

    : Op("Dequantize", check_single_output_args({input, scale, offset}))
    , m_type(type)
    , m_axes(axes)
{
    constructor_validate_and_infer_types();
}

void op::Dequantize::validate_and_infer_types()
{
    enum
    {
        INPUT,
        SCALE,
        OFFSET
    };

    NODE_VALIDATION_ASSERT(this, m_type.is_static()) << "Output element type must not be dynamic";

    NODE_VALIDATION_ASSERT(this, m_type.is_real()) << "Output element type (" << m_type
                                                   << ") must be a floating point type";

    element::Type quantized_type;

    NODE_VALIDATION_ASSERT(this,
                           element::Type::merge(quantized_type,
                                                get_input_element_type(INPUT),
                                                get_input_element_type(OFFSET)))
        << "Offset element type (" << get_input_element_type(OFFSET)
        << ") must match input element type (" << get_input_element_type(INPUT) << ")";

    NODE_VALIDATION_ASSERT(this, quantized_type.is_dynamic() || quantized_type.is_quantized())
        << "Offset/input element type (" << quantized_type << ") must be a quantized type";

    element::Type unquantized_type;

    NODE_VALIDATION_ASSERT(
        this, element::Type::merge(unquantized_type, get_input_element_type(SCALE), m_type))
        << "Scale element type (" << get_input_element_type(SCALE)
        << ") must match output element type (" << m_type << ")";

    PartialShape input_shape = get_input_partial_shape(0);
    Dimension input_rank = input_shape.rank();

    for (auto axis : m_axes)
    {
        NODE_VALIDATION_ASSERT(this, input_rank.is_dynamic() || axis < size_t(input_rank))
            << "Quantization axis (" << axis << ") must be less than input shape rank ("
            << input_rank << ")";
    }

    PartialShape scale_offset_shape = get_input_partial_shape(SCALE);

    NODE_VALIDATION_ASSERT(
        this, PartialShape::merge_into(scale_offset_shape, get_input_partial_shape(OFFSET)))
        << "Scale shape (" << get_input_partial_shape(SCALE) << ") and offset shape ("
        << get_input_partial_shape(OFFSET) << ") must match";

    NODE_VALIDATION_ASSERT(this, scale_offset_shape.rank().compatible(m_axes.size()))
        << "Scale/offset rank (" << scale_offset_shape.rank() << ") does not match the number of "
        << "quantization axes (" << m_axes.size() << ")";

    set_output_size(1);

    if (input_shape.rank().is_static() && scale_offset_shape.rank().is_static())
    {
        size_t i = 0;

        std::vector<Dimension> injected_scale_offset_dims;

        for (size_t j = 0; j < size_t(input_shape.rank()); j++)
        {
            if (m_axes.count(j) != 0)
            {
                injected_scale_offset_dims.push_back(scale_offset_shape[i++]);
            }
            else
            {
                injected_scale_offset_dims.push_back(Dimension::dynamic());
            }
        }

        PartialShape result_shape = input_shape;
        NODE_VALIDATION_ASSERT(
            this, PartialShape::merge_into(result_shape, PartialShape{injected_scale_offset_dims}))
            << "Scale/offset shape (" << scale_offset_shape << ") must match input shape ("
            << input_shape << ") at the quantization axes (" << m_axes << ")";
        set_output_type(0, unquantized_type, result_shape);
    }
    else
    {
        set_output_type(0, unquantized_type, PartialShape::dynamic());
    }
}

shared_ptr<Node> op::Dequantize::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Dequantize>(new_args.at(0), new_args.at(1), new_args.at(2), m_type, m_axes);
}

void op::Dequantize::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    throw ngraph_error("Forward-propagation-only operation");
}
