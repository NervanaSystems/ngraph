//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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
#include <numeric>
#include <sstream>

#include "graph.hpp"
#include "node.hpp"
#include "utils/common.hpp"

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

            static std::string concat_strings(
                const std::vector<std::reference_wrapper<const std::string>>& strings)
            {
                const auto concat_with_comma =
                    [](const std::string& accumulator,
                       std::reference_wrapper<const std::string> next_string) {
                        return accumulator + ", " + next_string.get();
                    };

                return std::accumulate(
                    strings.begin() + 1, strings.end(), strings.begin()->get(), concat_with_comma);
            }

            static std::string build_input_provenance_tag(const std::string& input_name,
                                                          const Shape& shape)
            {
                std::stringstream tag_builder;
                tag_builder << "<ONNX Input (" << input_name << ") " << shape << ">";
                return tag_builder.str();
            }

            static std::string build_op_provenance_tag(const Node& onnx_node)
            {
                const auto output_names = concat_strings(onnx_node.get_output_names());
                const auto node_name =
                    onnx_node.get_name().empty() ? "" : onnx_node.get_name() + " ";

                return std::string{"<ONNX " + onnx_node.op_type() + " (" + node_name + "-> " +
                                   output_names + ")>"};
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
                    auto ng_constant = tensor.get_ng_constant();
                    add_provenance_tag_to_initializer(tensor, ng_constant);
                    m_ng_node_cache.emplace(initializer_tensor.name(), std::move(ng_constant));
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

                const auto value_info = m_inputs.back();
                auto ng_node = value_info.get_ng_node(m_parameters, m_initializers, weights);
                add_provenance_tag_to_input(value_info, ng_node);
                m_ng_node_cache[input.name()] = std::move(ng_node);
            }

            // Process all graph outputs
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
                    // If a node from an unregistered domain is detected, try registering that
                    // domain
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

            NGRAPH_CHECK(unknown_operators.empty(),
                         "nGraph does not support the following ONNX operations: ",
                         detail::to_string(unknown_operators));

            // Process ONNX graph nodes, convert to nGraph nodes
            for (const auto& node_proto : m_graph_proto->node())
            {
                m_nodes.emplace_back(node_proto, *this);
                const Node& node{m_nodes.back()};

                NodeVector ng_nodes{node.get_ng_nodes()};
                // Iterate over the number of outputs for given node in graph.
                // Some of them may be optional and trimmed. See:
                // https://github.com/onnx/onnx/blob/master/docs/IR.md#optional-inputs-and-outputs
                for (std::size_t i{0}; i < node.get_outputs_size(); ++i)
                {
                    m_ng_node_cache[node.output(i)] = ng_nodes.at(i);
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

        NodeVector Graph::make_ng_nodes(const Node& onnx_node) const
        {
            const auto ng_node_factory =
                m_model->get_operator(onnx_node.op_type(), onnx_node.domain());

            const auto ng_node_vector = ng_node_factory(onnx_node);
            add_provenance_tags(onnx_node, ng_node_vector);

            return ng_node_vector;
        }

        void Graph::add_provenance_tag_to_initializer(
            const Tensor& tensor, std::shared_ptr<default_opset::Constant> node) const
        {
            const std::string tag =
                detail::build_input_provenance_tag(tensor.get_name(), tensor.get_shape());

            node->add_provenance_tag(tag);
        }

        void Graph::add_provenance_tag_to_input(const ValueInfo& input,
                                                std::shared_ptr<ngraph::Node> node) const
        {
            const std::string tag =
                detail::build_input_provenance_tag(input.get_name(), input.get_shape());

            node->add_provenance_tag(tag);
        }

        void Graph::add_provenance_tags(const Node& onnx_node,
                                        const NodeVector& ng_node_vector) const
        {
            const auto tag = detail::build_op_provenance_tag(onnx_node);
            const auto ng_inputs = onnx_node.get_ng_inputs();

            ngraph::traverse_nodes(
                ng_node_vector,
                [&tag](std::shared_ptr<ngraph::Node> ng_node) { ng_node->add_provenance_tag(tag); },
                false,
                ng_inputs);
        }
    } // namespace onnx_import

} // namespace ngraph
