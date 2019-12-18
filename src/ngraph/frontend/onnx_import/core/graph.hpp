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

#pragma once

#include <onnx/onnx_pb.h>
#include <string>
#include <vector>

#include "model.hpp"
#include "ngraph/op/parameter.hpp"
#include "operator_set.hpp"
#include "value_info.hpp"
#include "weight.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        class Graph
        {
        public:
            Graph(const onnx::GraphProto& proto, Model& model, const Weights& weights = {});

            const std::vector<Node>& get_nodes() const { return m_nodes; }
            const std::vector<ValueInfo>& get_inputs() const { return m_inputs; }
            const std::vector<ValueInfo>& get_outputs() const { return m_outputs; }
            NodeVector get_ng_outputs() const;
            const ParameterVector& get_ng_parameters() const { return m_parameters; }
            std::shared_ptr<ngraph::Node> get_ng_node_from_cache(const std::string& name) const
            {
                return m_ng_node_cache.at(name);
            }

            const std::string& get_name() const { return m_graph_proto->name(); }
            NodeVector make_ng_nodes(const Node& onnx_node) const
            {
                auto ng_node_vector =
                    m_model->get_operator(onnx_node.op_type(), onnx_node.domain())(onnx_node);
                for (auto& ng_node : ng_node_vector)
                {
                    std::string node_name =
                        onnx_node.get_name().empty() ? "unnamed node" : onnx_node.get_name();
                    std::string provenance_tag =
                        "<ONNX " + onnx_node.op_type() + " (" + node_name + ") >";
                    ng_node->add_provenance_tag(provenance_tag);

                    // FOR DEBBUG ONLY
                    auto tags = ng_node->get_provenance_tags();
                    for (auto tag : tags)
                    {
                         std::cout << tag << std::endl;
                    }
                }
                return ng_node_vector;
            }

        private:
            const onnx::GraphProto* m_graph_proto;
            std::vector<Node> m_nodes;
            std::vector<ValueInfo> m_inputs;
            std::vector<ValueInfo> m_outputs;
            ParameterVector m_parameters;
            std::map<std::string, std::shared_ptr<ngraph::Node>> m_ng_node_cache;
            std::map<std::string, Tensor> m_initializers;
            Model* m_model;
        };

        inline std::ostream& operator<<(std::ostream& outs, const Graph& graph)
        {
            return (outs << "<Graph: " << graph.get_name() << ">");
        }

    } // namespace onnx_import

} // namespace ngraph
