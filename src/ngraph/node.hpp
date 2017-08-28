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

#include <set>
#include <string>
#include <vector>

#include <iostream>

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

    protected:
        Node(const std::vector<Node::ptr>& arguments, ValueType::ptr type = nullptr)
            : TypedValueMixin(type)
            , m_arguments(arguments)
        {
            // Add this node as a user of each argument.
            for (auto node : m_arguments)
            {
                node->m_users.insert(node.get());
            }
        }

        virtual ~Node() {}

    public:
        /// A "one-liner" describing this node.
        virtual std::string description() const = 0;

        /// Propagate types and check arguments for consistency
        virtual void propagate_types() = 0;

        const std::vector<Node::ptr>& arguments() const { return m_arguments; }

        const std::multiset<Node*>& users() const { return m_users; }

        std::string name() const { return m_name; }
        void        name(const std::string& name) { m_name = name; }

        /**
         ** Return true if this has the same implementing class as call. This
         ** will be used by the pattern matcher when comparing a pattern
         ** graph against the graph.
         ** TODO: typeids are Node*, doc says they should be the actual classes.
         **/
         bool has_same_op(const Node::ptr& node) { return typeid(this) == typeid(node.get()); }
         

    protected:
        std::vector<Node::ptr> m_arguments;
        std::multiset<Node*>   m_users;
        std::string            m_name;
    };
}
