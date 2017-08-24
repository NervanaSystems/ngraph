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

#pragma once

#include <vector>
#include <set>

#include "ngraph/type.hpp"

namespace ngraph
{
    class Op;

    /**
     ** Nodes are the backbone of the graph of Value dataflow. Every node has
     ** zero or more nodes as arguments and one value, which is either a tensor
     ** view or a (possibly empty) tuple of values.
     **/
    class Node : public TypedValueMixin
    {
    public:

        using ptr = std::shared_ptr<Node>;

        Node(const std::vector<Node::ptr>& arguments, ValueType::ptr type = nullptr)
            : TypedValueMixin(type)
            , m_arguments(arguments)
        {
            // Add this node as a user of each argument.
            for(auto node : m_arguments){
                node->m_users.insert(node.get());
            }
        }

        const std::vector<Node::ptr> arguments() const { return m_arguments; }
        std::vector<Node::ptr>       arguments() { return m_arguments; }

        const std::multiset<Node*> users() const { return m_users; }
        std::multiset<Node*>       users() { return m_users; }

    protected:
        std::vector<Node::ptr> m_arguments;
        std::multiset<Node*> m_users;
    };
}
