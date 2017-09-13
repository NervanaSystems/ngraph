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

#include "ngraph/common.hpp"
#include "ngraph/type.hpp"

namespace ngraph
{
    class Op;

    namespace descriptor
    {
        class Input;
        class Output;
    }

    /// Nodes are the backbone of the graph of Value dataflow. Every node has
    /// zero or more nodes as arguments and one value, which is either a tensor
    /// view or a (possibly empty) tuple of values.
    class Node : public std::enable_shared_from_this<Node>
    {
    protected:
        Node(const Nodes& arguments, std::shared_ptr<ValueType> value_type = nullptr);
        Node()
            : Node({}, nullptr)
        {
        }

        Node(std::shared_ptr<ValueType> value_type)
            : Node({}, value_type)
        {
        }

        virtual ~Node();

    public:
        /// A "one-liner" describing this node.
        virtual std::string description() const = 0;

        /// Propagate types and check arguments for consistency
        virtual void propagate_types() = 0;

        /// Assign Input and Output vectors
        // This might later need to be virtual.
        void assign_tensors();

        const Nodes& get_arguments() const { return m_arguments; }
        void clear_arguments() { m_arguments.clear(); }

        const std::multiset<Node*>& users() const { return m_users; }

        std::string get_name() const { return m_name; }
        void set_name(const std::string& name) { m_name = name; }

        virtual std::string get_node_id() const = 0;

        /// Return true if this has the same implementing class as node. This
        /// will be used by the pattern matcher when comparing a pattern
        /// graph against the graph.
        bool is_same_op_type(const std::shared_ptr<Node>& node) const
        {
            return typeid(*this) == typeid(*node.get());
        }

        std::shared_ptr<ValueType>       get_value_type() { return m_value_type; }
        const std::shared_ptr<ValueType> get_value_type() const { return m_value_type; }

        void set_value_type(const element::Type& element_type, const Shape& shape)
        {
            m_value_type = std::make_shared<TensorViewType>(element_type, shape);
        }

        void set_value_type(const std::shared_ptr<ValueType>& value_type)
        {
            m_value_type = value_type;
        }

        // Set the value type if it has not already been set; otherwise, ensure that
        // value_type agrees with the value type that was set.
        // This is used when the framework specifies a value type for the value, and we
        // independently compute what we thing the value type should be from the arguments.
        void set_value_type_checked(const std::shared_ptr<ValueType>& value_type);

        bool is_op() const;
        bool is_parameter() const;

        size_t               get_instance_id() const { return m_instance_id; }
        friend std::ostream& operator<<(std::ostream&, const Node&);

        std::vector<descriptor::Input>&  get_inputs() { return m_inputs; }
        std::vector<descriptor::Output>& get_outputs() { return m_outputs; }

    protected:
        Nodes                           m_arguments;
        std::shared_ptr<ValueType>      m_value_type;
        std::multiset<Node*>            m_users;
        std::string                     m_name;
        size_t                          m_instance_id;
        static size_t                   m_next_instance_id;
        std::vector<descriptor::Input>  m_inputs;
        std::vector<descriptor::Output> m_outputs;
    };
}
