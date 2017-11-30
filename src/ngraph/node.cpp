// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include "ngraph/node.hpp"
#include <memory>
#include <typeindex>
#include <typeinfo>

#include "ngraph/autodiff/adjoints.hpp"
#include "ngraph/descriptor/primary_tensor_view.hpp"
#include "ngraph/ops/parameter.hpp"

using namespace std;
using namespace ngraph;

atomic<size_t> Node::m_next_instance_id(0);

Node::Node(const std::string& node_type, const std::vector<shared_ptr<Node>>& arguments)
    : m_node_type(node_type)
    , m_arguments(arguments)
    , m_instance_id(m_next_instance_id.fetch_add(1))
    , m_is_output(false)
{
    // Add this node as a user of each argument.
    size_t i = 0;
    size_t argno = 0;
    for (auto arg : m_arguments)
    {
        arg->m_users.insert(this);
        size_t arg_index = 0;
        for (descriptor::Output& output : arg->get_outputs())
        {
            m_inputs.emplace_back(this, i, argno, arg_index++, output);
            i++;
        }
        argno++;
    }
}

Node::~Node()
{
}

void Node::assert_value_type(const shared_ptr<const ValueType>& value_type) const
{
    if (*m_value_type != *value_type)
    {
        throw ngraph_error("Setting value type to a different ValueType");
    }
}

void Node::set_value_type_checked(const shared_ptr<const ValueType>& value_type)
{
    if (nullptr == m_value_type)
    {
        if (nullptr != value_type)
        {
            m_value_type = value_type;
            vector<std::shared_ptr<const TensorViewType>> tensor_view_types;
            m_value_type->collect_tensor_views(tensor_view_types);
            size_t i = 0;
            for (auto tvt : tensor_view_types)
            {
                auto tensor_view_descriptor = make_shared<descriptor::PrimaryTensorView>(
                    tvt,
                    ngraph::descriptor::Tensor::make_tensor_name(this, i),
                    is_output(),
                    is_parameter(),
                    is_constant());
                m_outputs.emplace_back(this, i, tensor_view_descriptor);
                i++;
            }
        }
    }
    else
    {
        if (*m_value_type != *value_type)
        {
            throw ngraph_error("Setting value type to a different ValueType");
        }
    }
}

std::shared_ptr<const ValueType> Node::get_value_type()
{
    return m_value_type;
}

const std::shared_ptr<const ValueType> Node::get_value_type() const
{
    return m_value_type;
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
    return m_is_output;
}

void Node::set_is_output()
{
    m_is_output = true;
    for (descriptor::Output& output : get_outputs())
    {
        output.get_tensor().set_is_output();
    }
}

bool Node::is_constant() const
{
    return false;
}

std::string Node::get_node_id() const
{
    stringstream ss;
    ss << description() << "_" << m_instance_id;
    return ss.str();
}

std::string Node::get_name() const
{
    string rc;
    if (m_name.empty())
    {
        rc = description() + "_" + to_string(m_instance_id);
    }
    else
    {
        rc = m_name;
    }
    return rc;
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

std::shared_ptr<Node> Node::backprop_node(const std::shared_ptr<Node>& x,
                                          const std::shared_ptr<Node>& c)
{
    auto adjoints_it = m_adjoint_map.find(c.get());
    if (adjoints_it == m_adjoint_map.end())
    {
        adjoints_it =
            m_adjoint_map.insert({c.get(), autodiff::Adjoints(shared_from_this(), c)}).first;
    }
    return adjoints_it->second.get(x);
}

std::shared_ptr<Function> Node::get_function() const
{
    return nullptr;
}

namespace ngraph
{
    ostream& operator<<(ostream& out, const Node& node)
    {
        auto parameter_tmp = dynamic_cast<const op::Parameter*>(&node);
        if (parameter_tmp)
        {
            out << "Parameter(" << parameter_tmp->get_node_id() << ")";
        }
        else
        {
            out << "Node(" << node.get_node_id() << ")";
        }
        return out;
    }
}
