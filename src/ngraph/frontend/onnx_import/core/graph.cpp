//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include <functional>
#include <set>

#include "graph.hpp"
#include "node.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace detail
        {
            static std::string to_string(
                const std::map<std::string, std::reference_wrapper<const onnx::NodeProto>>& map)
            {
                std::string result;
                for (auto it = std::begin(map); it != std::end(map); ++it)
                {
                    result += (it != std::begin(map) ? ", " : "") + it->first;
                }
                return result;
            }

            static std::string get_node_domain(const onnx::NodeProto& node_proto)
            {
                return (node_proto.domain().empty() ? "" : node_proto.domain());
            }

            /// \brief      Gets the operator represented by provided node unique identificator.
            ///
            /// \param[in]  node_proto  The node protobuf representation object.
            ///
            /// \note       The operator is uniquely identified by the tuple (domain, op_type,
            ///             since_version). The first two elements are stored in NodeProto object,
            ///             thus we use only them.
            ///
            /// \return     The unique identificator.
            ///
            static std::string get_op_domain_and_name(const onnx::NodeProto& node_proto)
            {
                std::string domain = get_node_domain(node_proto);
                return (domain.empty() ? "" : domain + ".") + node_proto.op_type();
            }
        } // namespace detail

        Graph::Graph(const onnx::GraphProto& graph_proto, Model& model, const Weights& weights)
            : m_graph_proto{&graph_proto}
            , m_model{&model}
        {
            // Process all initializers in the graph
            for (const auto& initializer_tensor : m_graph_proto->initializer())
            {
                if (initializer_tensor.has_name())
                {
                    Tensor tensor = Tensor{initializer_tensor};
                    m_initializers.emplace(initializer_tensor.name(), tensor);

                    // For each initializer, create a Constant node and store in cache
                    m_ng_node_cache.emplace(initializer_tensor.name(), tensor.get_ng_constant());
                }
            }

            // Process all ONNX graph inputs, convert them to nGraph nodes and store in cache
            for (const auto& input : m_graph_proto->input())
            {
                m_inputs.emplace_back(input);

                // Check if a Constant node was already created from an initializer
                if (m_ng_node_cache.count(input.name()) > 0)
                {
                    continue;
                }

                m_ng_node_cache[input.name()] =
                    m_inputs.back().get_ng_node(m_parameters, m_initializers, weights);
            }

            for (const auto& output : m_graph_proto->output())
            {
                m_outputs.emplace_back(output);
            }

            // Verify that ONNX graph contains only nodes of available operator types
            std::map<std::string, std::reference_wrapper<const onnx::NodeProto>> unknown_operators;
            for (const auto& node_proto : m_graph_proto->node())
            {
                if (!m_model->is_operator_available(node_proto))
                {
                    unknown_operators.emplace(detail::get_op_domain_and_name(node_proto),
                                              node_proto);
                    // Try adding missing domain
                    m_model->enable_opset_domain(detail::get_node_domain(node_proto));
                }
            }

            // Reverify wheter we still have any unavailable operators.
            auto it = std::begin(unknown_operators);
            while (it != std::end(unknown_operators))
            {
                if (m_model->is_operator_available(it->second))
                {
                    it = unknown_operators.erase(it);
                }
                else
                {
                    it++;
                }
            }

            NGRAPH_ASSERT(unknown_operators.empty()) << "unknown operations: "
                                                     << detail::to_string(unknown_operators);

            // Process ONNX graph nodes, convert to nGraph nodes
            for (const auto& node_proto : m_graph_proto->node())
            {
                m_nodes.emplace_back(node_proto, *this);
                const Node& node{m_nodes.back()};
                NodeVector ng_nodes{node.get_ng_nodes()};
                for (int i = 0; i < ng_nodes.size(); i++)
                {
                    m_ng_node_cache[node.output(i)] = ng_nodes[i];
                }
            }
        }

        NodeVector Graph::get_ng_outputs() const
        {
            NodeVector results;
            for (const auto& output : m_graph_proto->output())
            {
                results.emplace_back(get_ng_node_from_cache(output.name()));
            }
            return results;
        }

    } // namespace onnx_import

} // namespace ngraph
