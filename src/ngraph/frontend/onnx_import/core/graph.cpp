//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "graph.hpp"
#include "node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        Graph::Graph(const onnx::GraphProto& graph_proto)
            : m_graph_proto{&graph_proto}
        {
            for (const auto& tensor : m_graph_proto->initializer())
            {
                if (tensor.has_name())
                {
                    m_initializers.emplace(tensor.name(), Tensor{tensor});
                }
            }

            // Process all ONNX graph inputs, convert them to nGraph nodes and store in cache
            for (const auto& input : m_graph_proto->input())
            {
                m_inputs.emplace_back(input);
                m_ng_node_cache[input.name()] =
                    m_inputs.back().get_ng_node(m_parameters, m_initializers);
            }

            for (const auto& output : m_graph_proto->output())
            {
                m_outputs.emplace_back(output);
            }

            // Process ONNX graph nodes, convert to nGraph nodes
            for (const auto& node_proto : m_graph_proto->node())
            {
                m_nodes.emplace_back(node_proto, this);
                const Node& node{m_nodes.back()};
                NodeVector ng_nodes{node.get_ng_nodes()};
                for (int i = 0; i < ng_nodes.size(); i++)
                {
                    m_ng_node_cache[node.output(i)] = ng_nodes[i];
                }
            }
        }

    } // namespace onnx_import

} // namespace ngraph
