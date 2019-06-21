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

op::TensorIterator::TensorIterator(const OutputVector& body_inputs,
                                   const ParameterVector& body_parameters,
                                   const OutputVector& body_outputs,
                                   const OutputVector& tensor_iterator_outputs)
    : Op(body_inputs)
    , m_body_parameters(body_parameters)
    , m_body_outputs(body_outputs)
    , m_tensor_iterator_outputs(tensor_iterator_outputs)
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

const OutputVector& op::TensorIterator::get_body_outputs() const
{
    return m_body_outputs;
}

OutputVector& op::TensorIterator::get_body_outputs()
{
    return m_body_outputs;
}

void op::TensorIterator::set_body_outputs(const OutputVector& body_outputs)
{
    m_body_outputs = body_outputs;
}

const OutputVector& op::TensorIterator::get_tensor_iterator_outputs() const
{
    return m_tensor_iterator_outputs;
}

OutputVector& op::TensorIterator::get_tensor_iterator_outputs()
{
    return m_tensor_iterator_outputs;
}

void op::TensorIterator::set_tensor_iterator_outputs(const OutputVector& tensor_iterator_outputs)
{
    m_tensor_iterator_outputs = tensor_iterator_outputs;
}

std::shared_ptr<Node> op::TensorIterator::copy_with_new_args(const NodeVector& new_args) const
{
    auto result = make_shared<TensorIterator>();
    result->set_arguments(new_args);
    result->set_body_parameters(m_body_parameters);
    result->set_body_outputs(m_body_outputs);
    result->set_tensor_iterator_outputs(m_tensor_iterator_outputs);
    return move(result);
}
