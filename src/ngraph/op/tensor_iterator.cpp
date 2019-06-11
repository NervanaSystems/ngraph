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

using namespace std;
using namespace ngraph;

op::SliceOutput::SliceOutput(const Output<Node>& value,
                             size_t result_position,
                             std::ptrdiff_t axis,
                             std::ptrdiff_t start,
                             std::ptrdiff_t stride,
                             std::ptrdiff_t part_size,
                             std::ptrdiff_t end)
    : m_value(value)
    , m_result_position(result_position)
    , m_axis(axis)
    , m_start(start)
    , m_stride(stride)
    , m_part_size(part_size)
    , m_end(end)
{
}

const string op::TensorIterator::type_name{"TensorIterator"};

op::TensorIterator::TensorIterator()
{
}

op::TensorIterator::TensorIterator(const OutputVector& inputs,
                                   const ParameterVector& body_parameters,
                                   const OutputVector& initial_body_arguments,
                                   const OutputVector& body_arguments,
                                   const OutputVector& outputs,
                                   std::vector<SliceOutput> slice_outputs)
    : Op(inputs)
    , m_body_parameters(body_parameters)
    , m_initial_body_arguments(initial_body_arguments)
    , m_body_arguments(body_arguments)
    , m_outputs(outputs)
    , m_slice_outputs(slice_outputs)
{
}

const ParameterVector& op::TensorIterator::get_body_parameters() const
{
    return m_body_parameters;
}

ParameterVector& op::TensorIterator::get_body_parameters()
{
    return m_body_parameters;
}

void op::TensorIterator::set_body_parameters(const ParameterVector& body_parameters)
{
    m_body_parameters = body_parameters;
}

const OutputVector& op::TensorIterator::get_initial_body_arguments() const
{
    return m_initial_body_arguments;
}

OutputVector& op::TensorIterator::get_initial_body_arguments()
{
    return m_initial_body_arguments;
}

void op::TensorIterator::set_initial_body_arguments(const OutputVector& initial_body_arguments)
{
    m_initial_body_arguments = initial_body_arguments;
}

const OutputVector& op::TensorIterator::get_body_arguments() const
{
    return m_body_arguments;
}

OutputVector& op::TensorIterator::get_body_arguments()
{
    return m_body_arguments;
}

void op::TensorIterator::set_body_arguments(const OutputVector& body_arguments)
{
    m_body_arguments = body_arguments;
}

const OutputVector& op::TensorIterator::get_outputs() const
{
    return m_outputs;
}
OutputVector& op::TensorIterator::get_outputs()
{
    return m_outputs;
}

void op::TensorIterator::set_outputs(const OutputVector& outputs)
{
    m_outputs = outputs;
}

const std::vector<op::SliceOutput>& op::TensorIterator::get_slice_outputs() const
{
    return m_slice_outputs;
}

std::vector<op::SliceOutput>& op::TensorIterator::get_slice_outputs()
{
    return m_slice_outputs;
}

void op::TensorIterator::set_slice_outputs(const std::vector<op::SliceOutput>& slice_outputs)
{
    m_slice_outputs = slice_outputs;
}

std::shared_ptr<Node> op::TensorIterator::copy_with_new_args(const NodeVector& new_args) const
{
    auto result = make_shared<TensorIterator>(OutputVector{},
                                              m_body_parameters,
                                              m_initial_body_arguments,
                                              m_body_arguments,
                                              m_outputs,
                                              m_slice_outputs);
    result->set_arguments(new_args);
    return std::move(result);
}
