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

#include "type.hpp"
#include "common.hpp"

namespace ngraph
{
    class Op;

    /// Nodes are the backbone of the graph of Value dataflow. Every node has
    /// zero or more nodes as arguments and one value, which is either a tensor
    /// view or a (possibly empty) tuple of values.
    class Node : public TypedValueMixin
    {
    public:
        using ptr = std::shared_ptr<Node>;

    protected:
        Node(const Nodes& arguments, ValueType::ptr type = nullptr);

        virtual ~Node() {}
    public:
        /// A "one-liner" describing this node.
        virtual std::string description() const = 0;

        /// Propagate types and check arguments for consistency
        virtual void propagate_types() = 0;

        const Nodes& get_arguments() const { return m_arguments; }

        const std::multiset<Node*>& users() const { return m_users; }

        std::string get_name() const { return m_name; }
        void        set_name(const std::string& name) { m_name = name; }

        virtual std::string get_node_id() const = 0;

        /// Return true if this has the same implementing class as node. This
        /// will be used by the pattern matcher when comparing a pattern
        /// graph against the graph.
        bool is_same_op_type(const Node::ptr& node) const
        {
            return typeid(*this) == typeid(*node.get());
        }

        bool is_op() const;
        bool is_parameter() const;

        size_t               get_instance_id() const { return m_instance_id; }
        friend std::ostream& operator<<(std::ostream&, const Node&);

    protected:
        Nodes m_arguments;
        std::multiset<Node*>   m_users;
        std::string            m_name;
        size_t                 m_instance_id;
        static size_t          m_next_instance_id;
    };

    using node_ptr = std::shared_ptr<Node>;
}
