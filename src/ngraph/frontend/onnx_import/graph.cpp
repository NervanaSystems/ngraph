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

#include "graph.hpp"
#include "node.hpp"

using namespace ngraph;

onnx_import::Graph::Graph(const onnx::GraphProto& graph_proto)
    : m_graph_proto(graph_proto)
{
    // Process all ONNX graph inputs, convert them to nGraph nodes and store in cache
    for (const auto& input : m_graph_proto.input())
    {
        ValueInfo value_info = ValueInfo(input, this);
        m_inputs.emplace_back(value_info);

        if (value_info.has_initializer())
        {
            // Input values with initializers produce Constant Nodes
            m_ng_node_cache[input.name()] = value_info.get_ng_node();
        }
        else
        {
            // Other input values produce Parameters
            auto parameter = value_info.get_ng_parameter();
            m_ng_node_cache[input.name()] = parameter;
            m_parameters.emplace_back(parameter);
        }
    }

    for (const auto& output : m_graph_proto.output())
        m_outputs.emplace_back(output, this);

    // Process ONNX graph nodes, convert to nGraph nodes
    for (const auto& node : m_graph_proto.node())
    {
        Node node_wrapper = Node(node, this);
        m_nodes.emplace_back(node_wrapper);

        auto ng_nodes = node_wrapper.get_ng_nodes();
        for (int i = 0; i < ng_nodes.size(); i++)
        {
            m_ng_node_cache[node_wrapper.get_proto().output(i)] = ng_nodes[i];
        }
    }
}

std::ostream& onnx_import::operator<<(std::ostream& os, const Graph& wrapper)
{
    os << "<Graph: " << wrapper.m_graph_proto.name() << ">";
    return os;
}

const std::vector<onnx_import::Node>& onnx_import::Graph::get_nodes() const
{
    return m_nodes;
}

const std::vector<onnx_import::ValueInfo>& onnx_import::Graph::get_inputs() const
{
    return m_inputs;
}

const std::vector<onnx_import::ValueInfo>& onnx_import::Graph::get_outputs() const
{
    return m_outputs;
}

const onnx_import::Tensor onnx_import::Graph::get_initializer(std::string name) const
{
    for (int i = 0; i < m_graph_proto.initializer_size(); i++)
    {
        const onnx::TensorProto& initializer = m_graph_proto.initializer(i);

        if (initializer.name() == name)
        {
            return onnx_import::Tensor(initializer, this);
        }
    }

    throw ngraph::ngraph_error("Initializer not found: " + name);
}

const std::shared_ptr<ngraph::Node>
    onnx_import::Graph::get_ng_node_from_cache(std::string value_name)
{
    return m_ng_node_cache[value_name];
}

const ngraph::op::ParameterVector onnx_import::Graph::get_ng_parameters()
{
    return m_parameters;
}
