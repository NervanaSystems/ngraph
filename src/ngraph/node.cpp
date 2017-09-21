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

#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;

size_t Node::m_next_instance_id = 0;

Node::Node(const std::vector<shared_ptr<Node>>& arguments, shared_ptr<ValueType> value_type)
    : m_arguments(arguments)
    , m_value_type(value_type)
    , m_instance_id(m_next_instance_id++)
    , m_is_output(false)
{
    // Add this node as a user of each argument.
    for (auto node : m_arguments)
    {
        node->m_users.insert(this);
    }
}

Node::~Node()
{
}

void Node::set_value_type_checked(const shared_ptr<ValueType>& value_type)
{
    if (nullptr == m_value_type)
    {
        m_value_type = value_type;
    }
    else
    {
        if (*m_value_type != *value_type)
        {
            throw ngraph_error("Setting value type to a different ValueType");
        }
    }
}

void Node::assign_tensors()
{
    vector<std::shared_ptr<const TensorViewType>> tensor_view_types;
    get_value_type()->collect_tensor_views(tensor_view_types);
    size_t i = 0;
    for (auto tvt : tensor_view_types)
    {
        auto tensor_view_descriptor = make_shared<descriptor::PrimaryTensorView>(tvt, this, i);
        m_outputs.emplace_back(this, i, tensor_view_descriptor);
        i++;
    }

    i            = 0;
    size_t argno = 0;
    for (auto arg : get_arguments())
    {
        size_t arg_index = 0;
        for (descriptor::Output& output : arg->get_outputs())
        {
            m_inputs.emplace_back(this, i, argno, arg_index++, output);
            i++;
        }
        argno++;
    }
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
}

std::string Node::get_node_id() const
{
    stringstream ss;
    ss << description() << "_" << m_instance_id;
    return ss.str();
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
