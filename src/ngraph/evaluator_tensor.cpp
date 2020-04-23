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
#include "ngraph/node.hpp"
#include "ngraph/node_output.hpp"
#include "ngraph/type/element_type.hpp"

std::string ngraph::node_evaluation_failure_loc_string(const Node* node)
{
    std::stringstream ss;
    ss << "While evaluating node '" << *node << "'";
    return ss.str();
}

ngraph::EvaluatorTensor::~EvaluatorTensor()
{
}

ngraph::EvaluatorTensor::EvaluatorTensor(const element::Type& element_type,
                                         const PartialShape& partial_shape,
                                         bool is_allocated)
    : m_element_type(element_type)
    , m_partial_shape(partial_shape)
    , m_is_allocated(is_allocated)
{
}

ngraph::EvaluatorTensor::EvaluatorTensor(const Output<Node>& value, bool is_allocated)
    : EvaluatorTensor(value.get_element_type(), value.get_partial_shape(), is_allocated)
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

size_t ngraph::EvaluatorTensor::get_element_count() const
{
    return shape_size(get_partial_shape().get_shape());
}

size_t ngraph::EvaluatorTensor::get_size_in_bytes() const
{
    return get_element_type().size() * get_element_count();
}

void ngraph::EvaluatorTensor::set_element_type(Node* node, const element::Type& element_type)
{
    NODE_EVALUATION_CHECK(node,
                          m_element_type.is_dynamic() || m_element_type == element_type,
                          "Can not change a static element type");
    m_element_type = element_type;
}

void ngraph::EvaluatorTensor::set_is_allocated()
{
    m_is_allocated = true;
}

bool ngraph::EvaluatorTensor::get_is_allocated() const
{
    return m_is_allocated;
}

void ngraph::EvaluatorTensor::set_shape(Node* node, const Shape& shape)
{
    NODE_EVALUATION_CHECK(node,
                          PartialShape(shape).refines(m_partial_shape),
                          "Allocation shape ",
                          shape,
                          " must be compatible with the partial shape: ",
                          m_partial_shape);
    m_partial_shape = shape;
}

void ngraph::EvaluatorTensor::set_unary(Node* node, const EvaluatorTensorPtr& arg)
{
    set_element_type(node, arg->get_element_type());
    set_shape(node, arg->get_partial_shape().get_shape());
}

void ngraph::EvaluatorTensor::set_broadcast(Node* node,
                                            const op::AutoBroadcastSpec& autob,
                                            const EvaluatorTensorPtr& arg0,
                                            const EvaluatorTensorPtr& arg1)
{
    element::Type element_type = arg0->get_element_type();
    NODE_EVALUATION_CHECK(
        node,
        element::Type::merge(element_type, element_type, arg1->get_element_type()),
        "Argument element types are inconsistent.");
    set_element_type(node, element_type);

    PartialShape pshape = arg0->get_partial_shape();
    if (autob.m_type == op::AutoBroadcastType::NONE)
    {
        NODE_EVALUATION_CHECK(node,
                              PartialShape::merge_into(pshape, arg1->get_partial_shape()),
                              "Argument shapes are inconsistent.");
    }
    else if (autob.m_type == op::AutoBroadcastType::NUMPY ||
             autob.m_type == op::AutoBroadcastType::PDPD)
    {
        NODE_EVALUATION_CHECK(
            node,
            PartialShape::broadcast_merge_into(pshape, arg1->get_partial_shape(), autob),
            "Argument shapes are inconsistent.");
    }
    else
    {
        NODE_EVALUATION_CHECK(node, false, "Unsupported auto broadcast specification");
    }
    set_shape(node, pshape.get_shape());
}
