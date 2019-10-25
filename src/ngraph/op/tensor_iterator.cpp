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

#include "ngraph/op/tensor_iterator.hpp"
#include "ngraph/graph_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::TensorIterator::type_info;

constexpr DiscreteTypeInfo op::TensorIterator::SliceInputDescription::type_info;
constexpr DiscreteTypeInfo op::TensorIterator::BodyConnectionInputDescription::type_info;

constexpr DiscreteTypeInfo op::TensorIterator::BodyOutputDescription::type_info;
constexpr DiscreteTypeInfo op::TensorIterator::ConcatOutputDescription::type_info;

op::TensorIterator::TensorIterator(const OutputVector& values)
    : op::util::FusedOp(values)
{
}

op::TensorIterator::InputDescription::InputDescription(
    uint64_t input_index, const std::shared_ptr<Parameter>& body_parameter)
    : m_input_index(input_index)
    , m_body_parameter(body_parameter)
{
}

op::TensorIterator::SliceInputDescription::SliceInputDescription(
    uint64_t input_index,
    const std::shared_ptr<Parameter>& body_parameter,
    int64_t start,
    int64_t stride,
    uint64_t part_size,
    int64_t end,
    int64_t axis)
    : InputDescription(input_index, body_parameter)
    , m_start(start)
    , m_stride(stride)
    , m_part_size(part_size)
    , m_end(end)
    , m_axis(axis)
{
}

shared_ptr<op::TensorIterator::InputDescription>
    op::TensorIterator::SliceInputDescription::copy() const
{
    return make_shared<SliceInputDescription>(
        m_input_index, m_body_parameter, m_start, m_stride, m_part_size, m_end, m_axis);
}

op::TensorIterator::BodyConnectionInputDescription::BodyConnectionInputDescription(
    uint64_t input_index,
    const std::shared_ptr<Parameter>& body_parameter,
    const Output<Node>& body_value)
    : InputDescription(input_index, body_parameter)
    , m_body_value(body_value)
{
}

shared_ptr<op::TensorIterator::InputDescription>
    op::TensorIterator::BodyConnectionInputDescription::copy() const
{
    return make_shared<BodyConnectionInputDescription>(
        m_input_index, m_body_parameter, m_body_value);
}

op::TensorIterator::OutputDescription::OutputDescription(const Output<Node>& body_value,
                                                         uint64_t output_index)
    : m_body_value(body_value)
    , m_output_index(output_index)
{
}

op::TensorIterator::ConcatOutputDescription::ConcatOutputDescription(const Output<Node>& body_value,
                                                                     uint64_t output_index,
                                                                     int64_t start,
                                                                     int64_t stride,
                                                                     uint64_t part_size,
                                                                     int64_t end,
                                                                     int64_t axis)
    : OutputDescription(body_value, output_index)
    , m_start(start)
    , m_stride(stride)
    , m_part_size(part_size)
    , m_end(end)
    , m_axis(axis)
{
}

shared_ptr<op::TensorIterator::OutputDescription>
    op::TensorIterator::ConcatOutputDescription::copy() const
{
    return make_shared<ConcatOutputDescription>(
        m_body_value, m_output_index, m_start, m_stride, m_part_size, m_end, m_axis);
}

op::TensorIterator::BodyOutputDescription::BodyOutputDescription(const Output<Node>& body_value,
                                                                 uint64_t output_index,
                                                                 int64_t iteration)
    : OutputDescription(body_value, output_index)
    , m_iteration(iteration)
{
}

shared_ptr<op::TensorIterator::OutputDescription>
    op::TensorIterator::BodyOutputDescription::copy() const
{
    return make_shared<BodyOutputDescription>(m_body_value, m_output_index, m_iteration);
}

Input<Node> op::TensorIterator::input_for_value(const Output<Node>& value)
{
    for (auto input : inputs())
    {
        if (input.get_source_output() == value)
        {
            return input;
        }
    }
    auto input_index = get_input_size();
    set_argument(input_index, value);
    return Input<Node>(this, input_index);
}

void op::TensorIterator::set_sliced_input(const std::shared_ptr<op::Parameter>& body_parameter,
                                          const Output<Node>& value,
                                          int64_t start,
                                          int64_t stride,
                                          int64_t part_size,
                                          int64_t end,
                                          int64_t axis)
{
    m_input_descriptions.push_back(make_shared<SliceInputDescription>(
        input_for_value(value).get_index(), body_parameter, start, stride, part_size, end, axis));
}

void op::TensorIterator::set_initialized_input(const std::shared_ptr<Parameter>& body_parameter,
                                               const Output<Node>& initial_value,
                                               const Output<Node>& successive_value)
{
    m_input_descriptions.push_back(make_shared<BodyConnectionInputDescription>(
        input_for_value(initial_value).get_index(), body_parameter, successive_value));
}

Output<Node> op::TensorIterator::get_iter_value(const Output<Node>& body_value, int64_t iteration)
{
    auto output_index = get_output_size();
    m_output_descriptions.push_back(
        make_shared<BodyOutputDescription>(body_value, output_index, iteration));
    set_output_size(output_index + 1);
    return Output<Node>(shared_from_this(), output_index);
}

Output<Node> op::TensorIterator::get_concatenated_slices(const Output<Node>& body_value,
                                                         int64_t start,
                                                         int64_t stride,
                                                         int64_t part_size,
                                                         int64_t end,
                                                         int64_t axis)
{
    auto output_index = get_output_size();
    m_output_descriptions.push_back(make_shared<ConcatOutputDescription>(
        body_value, output_index, start, stride, part_size, end, axis));
    set_output_size(output_index + 1);
    return Output<Node>(shared_from_this(), output_index);
}

NodeVector op::TensorIterator::decompose_op() const
{
    // Stub
    return NodeVector{};
}

void op::TensorIterator::validate_and_infer_types()
{
    NODE_VALIDATION_CHECK(this,
                          get_input_size() == m_input_descriptions.size(),
                          "Number of inputs must be the same as number of input descriptions");

    size_t num_slice_input_desc = 0;
    for (auto input_description : m_input_descriptions)
    {
        if (auto slice_input_description = as_type_ptr<SliceInputDescription>(input_description))
        {
            NODE_VALIDATION_CHECK(
                this, num_slice_input_desc == 0, "Only one slice input description is allowed");
            num_slice_input_desc++;
            auto index = slice_input_description->m_input_index;
            auto body_param_partial_shape =
                slice_input_description->m_body_parameter->get_partial_shape();
            auto input_partial_shape = inputs().at(index).get_source_output().get_partial_shape();
            auto start = slice_input_description->m_start;
            auto part_size = slice_input_description->m_part_size;
            auto end = slice_input_description->m_end;
            if (end != -1)
            {
                m_num_iterations = end - start;
            }

            if (input_partial_shape.is_static())
            {
                auto input_shape = input_partial_shape.to_shape();
                auto axis = slice_input_description->m_axis;
                if (end == -1)
                {
                    // for simple RNN case where stride is the same as part_size
                    // when end is -1, we assume that we slice the input from "start" to the very
                    // end.
                    m_num_iterations = static_cast<size_t>(input_shape[axis]) / part_size - start;
                }

                if (body_param_partial_shape.is_static())
                {
                    auto body_param_shape = body_param_partial_shape.to_shape();
                    for (auto i = 0; i < input_shape.size(); i++)
                    {
                        if (i != axis)
                        {
                            NODE_VALIDATION_CHECK(
                                this,
                                input_shape[i] == body_param_shape[i],
                                "Iterator input is not compatible with body param");
                        }
                    }
                }
            }
        }
        else if (auto body_connection_input_description =
                     as_type_ptr<BodyConnectionInputDescription>(input_description))
        {
            auto index = body_connection_input_description->m_input_index;
            auto body_value_partial_shape =
                body_connection_input_description->m_body_value.get_partial_shape();
            auto body_param_partial_shape =
                body_connection_input_description->m_body_parameter->get_partial_shape();
            auto input_partial_shape = inputs().at(index).get_source_output().get_partial_shape();
            NODE_VALIDATION_CHECK(this,
                                  body_value_partial_shape.compatible(body_param_partial_shape),
                                  "Iterator successive value is not compatible with body param");
            NODE_VALIDATION_CHECK(this,
                                  input_partial_shape.compatible(body_param_partial_shape),
                                  "Iterator initial value is not compatible with body param");
        }
    }

    for (auto output_description : m_output_descriptions)
    {
        if (auto concat_output_description =
                as_type_ptr<ConcatOutputDescription>(output_description))
        {
            auto index = concat_output_description->m_output_index;
            auto body_value = concat_output_description->m_body_value;
            auto body_value_partial_shape = body_value.get_partial_shape();
            if (body_value_partial_shape.is_static())
            {
                auto body_value_shape = body_value_partial_shape.to_shape();
                auto start = concat_output_description->m_start;
                auto part_size = concat_output_description->m_part_size;
                auto end = concat_output_description->m_end;
                auto axis = concat_output_description->m_axis;
                Shape out_shape{body_value_shape};
                if (end != -1)
                {
                    // for simple RNN case where stride is the same as part_size
                    out_shape[axis] = (end - start) * part_size;
                    set_output_type(index, body_value.get_element_type(), out_shape);
                }
                else if (m_num_iterations != -1)
                {
                    // for simple RNN case where stride is the same as part_size
                    out_shape[axis] = m_num_iterations * part_size;
                    set_output_type(index, body_value.get_element_type(), out_shape);
                }
            }
        }
        else if (auto body_output_description =
                     as_type_ptr<BodyOutputDescription>(output_description))
        {
            auto index = body_output_description->m_output_index;
            auto body_value = body_output_description->m_body_value;
            set_output_type(index, body_value.get_element_type(), body_value.get_partial_shape());
        }
    }
}

std::shared_ptr<Node> op::TensorIterator::copy_with_new_args(const NodeVector& new_args) const
{
    auto op = make_shared<op::TensorIterator>(as_output_vector(new_args));
    for (auto& input_description : m_input_descriptions)
    {
        op->m_input_descriptions.push_back(input_description->copy());
    }
    for (auto& output_description : m_output_descriptions)
    {
        op->m_output_descriptions.push_back(output_description->copy());
    }
    return move(op);
}
