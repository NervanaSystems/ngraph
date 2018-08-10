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

#pragma once

#include <algorithm>
#include <ostream>
#include <string>

#include "ngraph/node_vector.hpp"

#include <onnx.pb.h>

#include "attribute.hpp"
#include "tensor.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace error
        {
            namespace node
            {
                struct UnknownAttribute : ngraph_error
                {
                    explicit UnknownAttribute(const std::string& node, const std::string& name)
                        : ngraph_error{"Node (" + node + "): unknown attribute \'" + name + "\'"}
                    {
                    }
                };

            } // namespace node

        } // namespace error

        // forward declaration
        class Graph;

        class Node
        {
        public:
            Node() = delete;
            Node(const onnx::NodeProto& node_proto, const Graph* graph)
                : m_node_proto{node_proto}
                , m_graph{graph}
                , m_attributes{std::begin(node_proto.attribute()), std::end(node_proto.attribute())}
                , m_output_names{std::begin(node_proto.output()), std::end(node_proto.output())}
            {
            }

            Node(Node&&) noexcept = default;
            Node(const Node&) = default;

            Node& operator=(Node&&) noexcept = delete;
            Node& operator=(const Node&) = delete;

            const std::vector<Attribute>& attributes() const { return m_attributes; }
            NodeVector get_ng_nodes() const;
            NodeVector get_ng_inputs() const;

            const std::string& op_type() const { return m_node_proto.op_type(); }
            const std::string& get_name() const { return m_node_proto.name(); }
            const std::vector<std::reference_wrapper<const std::string>>& get_output_names() const
            {
                return m_output_names;
            }
            const std::string& output(int index) const { return m_node_proto.output(index); }
            template <typename T>
            T get_attribute_value(const std::string& name, T default_value) const
            {
                auto it{std::find_if(
                    std::begin(m_attributes),
                    std::end(m_attributes),
                    [&](const Attribute& attribute) { return attribute.get_name() == name; })};
                if (it == std::end(m_attributes))
                {
                    return default_value;
                }
                return it->template get_value<T>();
            }

            template <typename T>
            T get_attribute_value(const std::string& name) const
            {
                auto it{std::find_if(
                    std::begin(m_attributes),
                    std::end(m_attributes),
                    [&](const Attribute& attribute) { return attribute.get_name() == name; })};
                if (it == std::end(m_attributes))
                {
                    throw error::node::UnknownAttribute{get_name(), name};
                }
                return it->template get_value<T>();
            }

        private:
            const onnx::NodeProto& m_node_proto;
            const Graph* m_graph;
            std::vector<Attribute> m_attributes;
            std::vector<std::reference_wrapper<const std::string>> m_output_names;
        };

        inline std::ostream& operator<<(std::ostream& outs, const Node& node)
        {
            return (outs << "<Node(" << node.op_type() << "): " << node.get_name() << ">");
        }

    } // namespace onnx_import

} // namespace ngraph
