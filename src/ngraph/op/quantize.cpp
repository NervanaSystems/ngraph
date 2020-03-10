//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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
#include "ngraph/shape_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Quantize::type_info;

op::Quantize::Quantize(const Output<Node>& input,
                       const Output<Node>& scale,
                       const Output<Node>& zero_point,
                       const element::Type& type,
                       const AxisSet& axes,
                       RoundMode round_mode)

    : Op({input, scale, zero_point})
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
        ZERO_POINT
    };

    NODE_VALIDATION_CHECK(this, m_type.is_static(), "Output element type must not be dynamic");

    NODE_VALIDATION_CHECK(
        this, m_type.is_quantized(), "Output element type (", m_type, ") must be a quantized type");

    element::Type unquantized_type;

    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(unquantized_type,
                                               get_input_element_type(INPUT),
                                               get_input_element_type(SCALE)),
                          "Scale element type (",
                          get_input_element_type(SCALE),
                          ") must match input element type (",
                          get_input_element_type(INPUT),
                          ")");

    NODE_VALIDATION_CHECK(this,
                          unquantized_type.is_dynamic() || unquantized_type.is_real(),
                          "Scale / input element type (",
                          unquantized_type,
                          ") must be a floating point number");

    element::Type quantized_type;

    NODE_VALIDATION_CHECK(
        this,
        element::Type::merge(quantized_type, get_input_element_type(ZERO_POINT), m_type),
        "Zero point element type (",
        get_input_element_type(ZERO_POINT),
        ") must match output element type (",
        m_type,
        ")");

    PartialShape input_shape = get_input_partial_shape(0);
    Dimension input_rank = input_shape.rank();

    for (auto axis : m_axes)
    {
        NODE_VALIDATION_CHECK(this,
                              input_rank.is_dynamic() || axis < input_rank.get_length(),
                              "Quantization axis (",
                              axis,
                              ") must be less than input shape rank (",
                              input_rank,
                              ")");
    }

    PartialShape scale_zero_point_shape = get_input_partial_shape(SCALE);

    NODE_VALIDATION_CHECK(
        this,
        PartialShape::merge_into(scale_zero_point_shape, get_input_partial_shape(ZERO_POINT)),
        "Scale shape (",
        get_input_partial_shape(SCALE),
        ") and zero point shape (",
        get_input_partial_shape(ZERO_POINT),
        ") must match");

    NODE_VALIDATION_CHECK(this,
                          scale_zero_point_shape.rank().compatible(m_axes.size()),
                          "Scale / zero point rank (",
                          scale_zero_point_shape.rank(),
                          ") does not match the number of ",
                          "quantization axes (",
                          m_axes.size(),
                          ")");

    set_output_size(1);

    if (input_shape.rank().is_static() && scale_zero_point_shape.rank().is_static())
    {
        size_t i = 0;

        vector<Dimension> injected_scale_zero_point_dims;

        for (size_t j = 0; j < input_shape.rank().get_length(); j++)
        {
            if (m_axes.count(j) != 0)
            {
                injected_scale_zero_point_dims.push_back(scale_zero_point_shape[i++]);
            }
            else
            {
                injected_scale_zero_point_dims.push_back(Dimension::dynamic());
            }
        }

        PartialShape result_shape = input_shape;
        NODE_VALIDATION_CHECK(
            this,
            PartialShape::merge_into(result_shape, PartialShape{injected_scale_zero_point_dims}),
            "Scale / zero point shape (",
            scale_zero_point_shape,
            ") must match input shape (",
            input_shape,
            ") at the quantization axes (",
            m_axes,
            ")");
        set_output_type(0, quantized_type, result_shape);
    }
    else
    {
        set_output_type(0, quantized_type, PartialShape::dynamic());
    }
}

shared_ptr<Node> op::Quantize::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Quantize>(
        new_args.at(0), new_args.at(1), new_args.at(2), m_type, m_axes, m_round_mode);
}

void op::Quantize::generate_adjoints(autodiff::Adjoints& /* adjoints */,
                                     const OutputVector& /* deltas */)
{
    throw ngraph_error("Forward-propagation-only operation");
}
