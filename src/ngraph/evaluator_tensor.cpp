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

#include "ngraph/evaluator_tensor.hpp"
#include "ngraph/node_output.hpp"
#include "ngraph/type/element_type.hpp"

#if 0
ngraph::EvaluatorTensor::~EvaluatorTensor()
{
}

ngraph::EvaluatorTensor::EvaluatorTensor(const element::Type& element_type,
                                         const PartialShape& partial_shape)
    : m_element_type(element_type)
    , m_partial_shape(partial_shape)
{
}

ngraph::EvaluatorTensor::EvaluatorTensor(const Output<Node>& value)
    : EvaluatorTensor(value.get_element_type(), value.get_partial_shape())
{
}

const ngraph::element::Type& ngraph::EvaluatorTensor::get_element_type() const
{
    return m_element_type;
}

const ngraph::PartialShape& ngraph::EvaluatorTensor::get_partial_shape() const
{
    return m_partial_shape;
}

const ngraph::Shape ngraph::EvaluatorTensor::get_shape() const
{
    return m_partial_shape.get_shape();
}

size_t ngraph::EvaluatorTensor::get_element_count()
{
    return shape_size(get_partial_shape().get_shape());
}

size_t ngraph::EvaluatorTensor::get_size_in_bytes()
{
    return get_element_type().size() * get_element_count();
}
#endif
