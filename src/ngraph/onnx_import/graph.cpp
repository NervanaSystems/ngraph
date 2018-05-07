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
    for (const auto& node : m_graph_proto.node())
        m_nodes.emplace_back(node, this);

    for (const auto& value : m_graph_proto.value_info())
        m_values.emplace_back(value, this);
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

const std::vector<onnx_import::ValueInfo>& onnx_import::Graph::get_values() const
{
    return m_values;
}
