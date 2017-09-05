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

#include "node.hpp"
#include "op.hpp"

size_t ngraph::Node::m_next_instance_id = 0;

ngraph::Node::Node(const std::vector<std::shared_ptr<Node>>& arguments,
                   std::shared_ptr<ValueType>                value_type)
    : m_arguments(arguments)
    , m_value_type(value_type)
    , m_instance_id(m_next_instance_id++)
{
    // Add this node as a user of each argument.
    for (auto node : m_arguments)
    {
        node->m_users.insert(this);
    }
}

bool ngraph::Node::is_op() const
{
    return dynamic_cast<const ngraph::Op*>(this) != nullptr;
}

bool ngraph::Node::is_parameter() const
{
    return dynamic_cast<const ngraph::op::Parameter*>(this) != nullptr;
}

namespace ngraph
{
    std::ostream& operator<<(std::ostream& out, const ngraph::Node& node)
    {
        auto op_tmp        = dynamic_cast<const ngraph::Op*>(&node);
        auto parameter_tmp = dynamic_cast<const ngraph::Op*>(&node);
        if (op_tmp)
        {
            out << "Op(" << op_tmp->get_node_id() << ")";
        }
        else if (parameter_tmp)
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
