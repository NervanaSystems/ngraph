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

#include "ngraph/node.hpp"
#include <memory>
#include <typeindex>
#include <typeinfo>

#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/descriptor/layout/tensor_view_layout.hpp"
#include "ngraph/descriptor/primary_tensor_view.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/placement.hpp"

#if not defined(EIGEN_MPL2_ONLY)
#error("The flag `EIGEN_MPL2_ONLY` must be defined");
#endif

using namespace std;
using namespace ngraph;

atomic<size_t> Node::m_next_instance_id(0);

Node::Node(const std::string& node_type, const NodeVector& arguments)
    : m_node_type(node_type)
    , m_instance_id(m_next_instance_id.fetch_add(1))
    , m_unique_name(description() + "_" + to_string(m_instance_id))
{
    // Add this node as a user of each argument.
    size_t i = 0;
    for (auto arg : arguments)
    {
        for (descriptor::Output& output : arg->get_outputs())
        {
            m_inputs.emplace_back(this, i++, output);
        }
    }
}

void Node::set_value_type_checked(const element::Type& element_type, const Shape& shape)
{
    if (m_outputs.size() == 0)
    {
        add_output(element_type, shape);
    }
    if (element_type != get_element_type() || shape != get_shape())
    {
        throw ngraph_error("Setting value type to a different ValueType");
    }
}

void Node::add_output(const element::Type& element_type, const Shape& shape)
{
    shared_ptr<TensorViewType> tensor_view_type = make_shared<TensorViewType>(element_type, shape);
    size_t i = m_outputs.size();
    auto tensor_view_descriptor = make_shared<descriptor::PrimaryTensorView>(
        tensor_view_type,
        ngraph::descriptor::Tensor::make_tensor_name(this, i),
        false,
        is_parameter(),
        is_constant());
    m_outputs.emplace_back(this, i, tensor_view_descriptor);
}

void Node::set_value_type_checked(const shared_ptr<const TensorViewType>& value_type)
{
    set_value_type_checked(value_type->get_element_type(), value_type->get_shape());
}

std::deque<descriptor::Output>& Node::get_outputs()
{
    return m_outputs;
}

const std::deque<descriptor::Output>& Node::get_outputs() const
{
    return m_outputs;
}

bool Node::is_parameter() const
{
    return dynamic_cast<const op::Parameter*>(this) != nullptr;
}

bool Node::is_output() const
{
    return false;
}

bool Node::is_constant() const
{
    return false;
}

const std::string& Node::get_friendly_name() const
{
    if (m_name.empty())
    {
        return m_unique_name;
    }
    return m_name;
}

const std::string& Node::get_name() const
{
    return m_unique_name;
}

void Node::set_name(const string& name)
{
    if (m_name.empty())
    {
        m_name = name;
    }
    else
    {
        throw ngraph_error("Node name may be set exactly once");
    }
}

Placement Node::get_placement() const
{
    return m_placement;
}

void Node::set_placement(Placement placement)
{
    m_placement = placement;
}

std::shared_ptr<Node> Node::get_input_op(size_t index)
{
    for (auto& i : get_inputs())
    {
        if (i.get_output().get_node()->get_outputs().size() != 1)
        {
            throw "get_input_op called on an argument w/ multiple outputs";
        }
    }
    return m_inputs.at(index).get_output().get_node();
}

Node::~Node()
{
    for (auto& input : m_inputs)
    {
        input.get_output().remove_input(&input);
    }
}

NodeVector Node::get_input_ops()
{
    NodeVector result;
    for (auto& i : get_inputs())
    {
        {
            result.push_back(i.get_output().get_node());
        }
    }
    return result;
}

std::vector<std::shared_ptr<Function>> Node::get_functions() const
{
    return std::vector<std::shared_ptr<Function>>{};
}

namespace ngraph
{
    ostream& operator<<(ostream& out, const Node& node)
    {
        auto parameter_tmp = dynamic_cast<const op::Parameter*>(&node);
        if (parameter_tmp)
        {
            out << "Parameter(" << parameter_tmp->get_name() << ")";
        }
        else
        {
            out << "Node(" << node.get_name() << ")";
        }
        return out;
    }
}

size_t Node::get_output_size() const
{
    return m_outputs.size();
}

const element::Type& Node::get_output_element_type(size_t i) const
{
    return m_outputs.at(i).get_element_type();
}

const element::Type& Node::get_element_type() const
{
    if (get_output_size() != 1)
    {
        throw ngraph_error("get_element_type() must be called on a node with exactly one output.");
    }
    return get_output_element_type(0);
}

const Shape& Node::get_output_shape(size_t i) const
{
    return m_outputs.at(i).get_shape();
}

const Shape& Node::get_shape() const
{
    if (get_output_size() != 1)
    {
        throw ngraph_error("get_shape() must be called on a node with exactly one output.");
    }
    return get_output_shape(0);
}

shared_ptr<descriptor::TensorView> Node::get_output_tensor_view(size_t i) const
{
    return m_outputs.at(i).get_tensor_view();
}

shared_ptr<descriptor::TensorView> Node::get_output_tensor_view() const
{
    if (get_output_size() != 1)
    {
        throw ngraph_error(
            "get_output_tensor_view() must be called on a node with exactly one output.");
    }
    return get_output_tensor_view(0);
}

const std::set<descriptor::Input*>& Node::get_output_inputs(size_t i) const
{
    return m_outputs.at(i).get_inputs();
}

descriptor::Tensor& Node::get_output_tensor(size_t i) const
{
    return m_outputs.at(i).get_tensor();
}

descriptor::Tensor& Node::get_output_tensor() const
{
    if (get_output_size() != 1)
    {
        throw ngraph_error("get_output_tensor() must be called on a node with exactly one output.");
    }
    return get_output_tensor(0);
}

size_t Node::get_input_size() const
{
    return m_inputs.size();
}

const element::Type& Node::get_input_element_type(size_t i) const
{
    return m_inputs.at(i).get_element_type();
}

const Shape& Node::get_input_shape(size_t i) const
{
    return m_inputs.at(i).get_shape();
}

bool Node::has_same_type(std::shared_ptr<const Node> node) const
{
    if (get_output_size() != node->get_output_size())
    {
        return false;
    }
    for (size_t i = 0; i < get_output_size(); ++i)
    {
        if (get_output_element_type(i) != node->get_output_element_type(i) ||
            get_output_shape(i) != node->get_output_shape(i))
        {
            return false;
        }
    }
    return true;
}

descriptor::Input* Node::get_input_from(const shared_ptr<Node>& src)
{
    for (size_t i = 0; i < this->get_input_size(); ++i)
    {
        if (this->get_input_op(i) == src)
        {
            return &(this->get_inputs().at(i));
        }
    }
    throw ngraph_error("Error: src is not one of self's input Node");
}

descriptor::Output* Node::get_output_to(const shared_ptr<Node>& dst)
{
    for (size_t i = 0; i < dst->get_input_size(); ++i)
    {
        if (dst->get_input_op(i).get() == this)
        {
            return &(dst->get_inputs().at(i).get_output());
        }
    }
    throw ngraph_error("Error: dst is not one of self's output Node");
}

NodeVector Node::get_users() const
{
    NodeVector result;

    for (size_t i = 0; i < get_output_size(); ++i)
    {
        for (auto input : get_output_inputs(i))
        {
            result.push_back(input->get_node());
        }
    }

    return result;
}
