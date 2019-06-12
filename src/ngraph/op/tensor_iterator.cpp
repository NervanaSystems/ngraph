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

const string op::TensorIterator::type_name{"TensorIterator"};

op::TensorIterator::TensorIterator()
{
}

op::TensorIterator::TensorIterator(const ParameterVector& body_parameters,
                                   const OutputVector& initial_body_arguments,
                                   const OutputVector& body_arguments,
                                   const OutputVector& outputs)
    : m_body_parameters(body_parameters)
    , m_initial_body_arguments(initial_body_arguments)
    , m_body_arguments(body_arguments)
    , m_outputs(outputs)
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

std::shared_ptr<Node> op::TensorIterator::copy_with_new_args(const NodeVector& new_args) const
{
    NGRAPH_CHECK(new_args.size() == 0, "TensorIterator has no arguments");
    return make_shared<TensorIterator>(
        m_body_parameters, m_initial_body_arguments, m_body_arguments, m_outputs);
}
