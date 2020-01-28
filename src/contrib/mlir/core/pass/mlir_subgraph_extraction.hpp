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

// NOTE: This file follows nGraph format style and naming convention since it
// exposes a public API to the rest of nGraph codebase.

#pragma once

#include <mutex>
#include "ngraph/pass/pass.hpp"
namespace ngraph
{
    namespace pass
    {
        /// This pass creates CompiledKernel ops enclosing maximal sub-graphs of ops that are
        /// supported by MLIR
        class MLIRSubgraphExtractionPass : public ngraph::pass::FunctionPass
        {
            using NodeSet = std::unordered_set<std::shared_ptr<Node>>;

            class MLIRSubgraph
            {
            private:
                static int get_new_graph_id() { return m_curr_graph_id++; }
                /// Create a sub-graph with a new ID.
                MLIRSubgraph(MLIRSubgraphExtractionPass* pass)
                    : m_graph_id(MLIRSubgraph::get_new_graph_id())
                    , m_pass(*pass)
                {
                }

            public:
                /// Factory method to creates a new sub-graph with unique ID
                static MLIRSubgraph create(MLIRSubgraphExtractionPass* pass)
                {
                    // mutex on global graph ID
                    std::lock_guard<std::mutex> lock(pass->m_subgraph_mutex);
                    return MLIRSubgraph(pass);
                }
                /// Get sub-graph id
                int get_id() const { return m_graph_id; }
                /// Get all nodes in the sub-graph.
                NodeVector& get_nodes() { return m_nodes; }
                /// Get input nodes. Predecessors to head nodes.
                NodeVector& get_inputs() { return m_input_node_vector; }
                /// Get output nodes. Nodes in the sub-graph with edges to external nodes.
                NodeVector& get_outputs() { return m_output_nodes; }
                /// Add a list of input nodes to the sub-graph.
                void add_inputs(NodeVector& inputs);
                /// Add a list of output nodes to the sub-graph.
                void add_outputs(NodeVector& outputs);
                /// Add one node to the sub-graph.
                void add_node(std::shared_ptr<Node> node);

            private:
                // Unique ID for this sub-graph.
                int m_graph_id;
                // Actual nodes of the sub-graph
                NodeVector m_nodes;
                // Predecessor to head nodes in the sub-graph. Both containers have the same
                // elements. Set is only used for efficient look-up operations.
                NodeVector m_input_node_vector;
                NodeSet m_input_node_set;

                NodeVector m_output_nodes;
                MLIRSubgraphExtractionPass& m_pass;
                static int m_curr_graph_id;
            };
            friend class MLIRSubgraph;

        public:
            bool run_on_function(std::shared_ptr<Function> func) override;
            /// Checks if an ngraph node is supported by MLIR backend
            bool is_supported_mlir_op(std::shared_ptr<Node> node);
            /// Get the sub-graph ID that a node belongs to
            int get_subgraph_id(std::shared_ptr<Node> node)
            {
                auto it = m_node_to_graph.find(node);
                return (it == m_node_to_graph.end()) ? -1 : it->second;
            }
            /// Get sub-graph by ID
            MLIRSubgraph& get_subgraph(int id)
            {
                auto it = m_id_to_graph.find(id);
                NGRAPH_CHECK(it != m_id_to_graph.end(), "Cannot find subgraph with ID: ", id);
                return it->second;
            }
            /// Stores a sub-graph in the map
            void add_subgraph(MLIRSubgraph& sg) { m_id_to_graph.emplace(sg.get_id(), sg); }
        private:
            void build_subgraphs(std::shared_ptr<Function> func);
            NodeVector build_ck_nodes(std::shared_ptr<Function> func);
            void process_supported_op(std::shared_ptr<ngraph::Node> node, int current_subgraph_id);

            void sanity_check(std::shared_ptr<Function> func, NodeVector& ck_nodes);
            void clean_up();
            static const std::set<ngraph::Node::type_info_t>& getSupportedOps();

        private:
            using IDGraphMap = std::unordered_map<int, MLIRSubgraph>;
            using NodeGraphMap = std::unordered_map<std::shared_ptr<Node>, int>;
            IDGraphMap m_id_to_graph;
            NodeGraphMap m_node_to_graph;
            // Mutex over sub-graph IDs
            std::mutex m_subgraph_mutex;
        };
    }
}
