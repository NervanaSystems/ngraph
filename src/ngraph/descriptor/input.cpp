/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/node.hpp"
#include "ngraph/type/element_type.hpp"

using namespace ngraph;
using namespace descriptor;

Input::Input(Node* node, size_t index, Output& output)
    : m_node(node)
    , m_index(index)
    , m_output(&output)
{
    m_src_node = std::shared_ptr<Node>(output.get_node());
    output.add_input(this);
}

void Input::replace_output(Output& new_output)
{
    m_output->remove_input(this);
    new_output.add_input(this);
    m_output = &new_output;
    m_src_node = std::shared_ptr<Node>(new_output.get_node());
}

void Input::replace_output(std::shared_ptr<Node> node, size_t i)
{
    replace_output(node->m_outputs.at(i));
}

std::shared_ptr<Node> Input::get_node()
{
    return m_node->shared_from_this();
}

const Tensor& Input::get_tensor() const
{
    return m_output->get_tensor();
}

Tensor& Input::get_tensor()
{
    return m_output->get_tensor();
}

std::shared_ptr<const TensorView> Input::get_tensor_view() const
{
    return m_output->get_tensor_view();
}

std::shared_ptr<TensorView> Input::get_tensor_view()
{
    return m_output->get_tensor_view();
}

std::shared_ptr<const TensorViewType> Input::get_tensor_view_type() const
{
    return m_output->get_tensor_view()->get_tensor_view_type();
}

const Shape& Input::get_shape() const
{
    return get_tensor_view_type()->get_shape();
}

const element::Type& Input::get_element_type() const
{
    return get_tensor_view_type()->get_element_type();
}
