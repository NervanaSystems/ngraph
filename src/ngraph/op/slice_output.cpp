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

#include "ngraph/op/slice_output.hpp"

using namespace std;
using namespace ngraph;

const string op::SliceOutput::type_name("SliceOutput");

op::SliceOutput::SliceOutput(const Output<Node>& value,
                             std::ptrdiff_t axis,
                             std::ptrdiff_t start,
                             std::ptrdiff_t stride,
                             std::ptrdiff_t part_size,
                             std::ptrdiff_t end)
    : Op({value})
    , m_axis(axis)
    , m_start(start)
    , m_stride(stride)
    , m_part_size(part_size)
    , m_end(end)
{
    constructor_validate_and_infer_types();
}

void op::SliceOutput::validate_and_infer_types()
{
    // For now, keep things simple; no padding, forwards stepping, etc.
    NODE_VALIDATION_CHECK(this, 0 <= m_axis, "Negative axis is not supported.");
    NODE_VALIDATION_CHECK(this, 0 <= m_start, "Negative start is not supported.");
    NODE_VALIDATION_CHECK(this, 0 <= m_end, "Negative end is not supported.");
    NODE_VALIDATION_CHECK(this, 1 <= m_part_size, "Part size must be positive.");
    NODE_VALIDATION_CHECK(this, 1 <= m_stride, "Stride must be positive.");

    PartialShape result_shape;
    const PartialShape& value_shape = get_input_partial_shape(0);

    if (value_shape.is_dynamic())
    {
        result_shape = PartialShape::dynamic();
    }
    else
    {
        vector<Dimension> value_dimensions = static_cast<vector<Dimension>>(value_shape);
        Dimension sequence_dimension = value_dimensions[0];
        value_dimensions.erase(value_dimensions.begin());

        NODE_VALIDATION_CHECK(this,
                              m_axis < value_dimensions.size(),
                              "Axis is not within the sequence element shape.");
        Dimension value_dimension = value_dimensions.at(m_axis);

        if (sequence_dimension.is_dynamic() || value_dimension.is_dynamic())
        {
            // Can't tell how long the result is
            value_dimension = Dimension();
        }
        else
        {
            size_t sequence_size = static_cast<size_t>(sequence_dimension);
            size_t value_size = static_cast<size_t>(value_dimension);
            NODE_VALIDATION_CHECK(
                this, m_start + m_part_size - 1 <= sequence_size, "start part is out of range.");
            NODE_VALIDATION_CHECK(
                this, m_end + m_part_size - 1 <= sequence_size, "end part is out of range.");
            size_t number_slices = (m_end - m_start) / m_stride;
            sequence_dimension = Dimension(number_slices * m_part_size * value_size);
        }
        value_dimensions[m_axis] = sequence_dimension;
        result_shape = PartialShape(value_dimensions);
    }
    set_output_size(1);
    set_output_type(0, get_input_element_type(0), result_shape);
}

shared_ptr<Node> op::SliceOutput::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<SliceOutput>(new_args.at(0), m_axis, m_start, m_stride, m_part_size, m_end);
}

ptrdiff_t op::SliceOutput::get_axis() const
{
    return m_axis;
}

void op::SliceOutput::set_axis(ptrdiff_t axis)
{
    m_axis = axis;
}
ptrdiff_t op::SliceOutput::get_start() const
{
    return m_start;
}
void op::SliceOutput::set_start(ptrdiff_t start)
{
    m_start = start;
}
ptrdiff_t op::SliceOutput::get_stride() const
{
    return m_stride;
}
void op::SliceOutput::set_stride(ptrdiff_t stride)
{
    m_stride = stride;
}
ptrdiff_t op::SliceOutput::get_part_size() const
{
    return m_part_size;
}
void op::SliceOutput::set_part_size(ptrdiff_t part_size)
{
    m_part_size = part_size;
}
ptrdiff_t op::SliceOutput::get_end() const
{
    return m_end;
}
void op::SliceOutput::set_end(ptrdiff_t end)
{
    m_end = end;
}
