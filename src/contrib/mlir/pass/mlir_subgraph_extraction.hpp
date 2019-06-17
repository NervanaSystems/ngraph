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

#include "ngraph/pass/pass.hpp"

namespace ngraph
{
    namespace pass
    {
        /// This pass creates CompiledKernel ops enclosing sub-graphs that will be compiled and
        /// executed by MLIR.
        class MLIRSubgraphExtractionPass : public ngraph::pass::FunctionPass
        {
        private:
            class MLIRSubgraph
            {
            private:
                // unique ID for this sub-graph. 
                int m_graph_id;
                // actual nodes of the sub-graph
                NodeVector m_nodes;
                // predecessor to nodes in the sub-graph. 
                NodeVector m_input_nodes;

            private:
                static int get_new_graph_id()
                {
                    static int graph_id = 0;
                    return graph_id++;
                }

            public:
                /// create a sub-graph with a new ID
                MLIRSubgraph() 
                {
                    m_graph_id = MLIRSubgraph::get_new_graph_id();
                }
                /// create a sub-graph with a specific ID
                MLIRSubgraph(int graph_id) : m_graph_id(graph_id) {}

                int get_id() const { return m_graph_id; }
                NodeVector& get_nodes() { return m_nodes; }
                NodeVector& get_input_nodes() { return m_input_nodes; }
                void add_inputs(NodeVector &inputs);
                void add_node(std::shared_ptr<Node> node);
                /// Merges other sub-graph into this sub-graph.
                void merge(MLIRSubgraph& other);
            };

        public:
            MLIRSubgraphExtractionPass() {}
            bool run_on_function(std::shared_ptr<Function> func) override;
            /// Checks if an ngraph node is supported by MLIR backend
            bool is_supported_mlir_op(std::shared_ptr<Node> node);
            int get_subgraph(std::shared_ptr<Node> node)
            {
                auto it = m_node_to_graph.find(node);
                return (it == m_node_to_graph.end()) ? -1 : it->second;
            }
            MLIRSubgraph& get_subgraph(int id)
            {
                auto it = m_id_to_graph.find(id);
                NGRAPH_CHECK(it != m_id_to_graph.end(), "Cannot find subgraph with ID: ", id);
                return it->second;
            }
            // Stores a sub-graph in the map and associates its nodes to it
            void add_subgraph(MLIRSubgraph sub_graph)
            {
                m_id_to_graph.emplace(sub_graph.get_id(), sub_graph);
                for (auto node : sub_graph.get_nodes())
                {
                    m_node_to_graph.emplace(node, sub_graph.get_id());
                }
            }
        private:
            using IDGraphMap = std::unordered_map<int, MLIRSubgraph>;
            using NodeGraphMap = std::unordered_map<std::shared_ptr<Node>, int>;
            IDGraphMap m_id_to_graph;
            NodeGraphMap m_node_to_graph;
        private:
            static const std::set<std::type_index> m_supported_ops;
        };
    }
}
