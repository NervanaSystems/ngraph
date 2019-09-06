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
                NodeSet& get_nodes() { return m_nodes; }
                /// Get input nodes. Predecessors to head nodes.
                NodeSet& get_inputs() { return m_input_nodes; }
                /// Get output nodes. Nodes in the sub-graph with edges to external nodes.
                NodeSet& get_outputs() { return m_output_nodes; }
                /// Add a list of input nodes to the sub-graph.
                template <typename T>
                void add_inputs(T& inputs);
                /// Add a list of output nodes to the sub-graph.
                template <typename T>
                void add_outputs(T& outputs);
                /// Merges sub-graph (other) into this sub-graph. other will be destroyed.
                void merge(MLIRSubgraph& other);
                /// Add one node to the sub-graph.
                void add_node(std::shared_ptr<Node> node);

            private:
                // Unique ID for this sub-graph.
                int m_graph_id;
                // Actual nodes of the sub-graph
                NodeSet m_nodes;
                // Predecessor to head nodes in the sub-graph.
                NodeSet m_input_nodes;
                NodeSet m_output_nodes;
                MLIRSubgraphExtractionPass& m_pass;
                static int m_curr_graph_id;
            };
            friend class MLIRSubgraph;

        public:
            MLIRSubgraphExtractionPass();
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
            /// Checks if adding a node to an extracted sub-graph will cause a DAG cycle
            /// inputs: the list of input nodes outside sub-graphs to the node we want to add.
            /// subgraph_ids: the sub-graphs the predecessor nodes belong to.
            /// It traverses backwards from all input nodes and checks if we left the to-be-merged
            /// sub-graphs and entered again. If so, we have a cycle.
            ///
            /// Example:
            /// A(1)
            /// |   \
            /// B(1) C
            /// |  /
            /// D
            /// we want to add D to sub-graph 1. C is an input to D. sugraph_ids are 1
            /// we traverse backwards C->A(1) and find 1, then we cannot add D since we will form a
            /// cycle
            bool check_cycles(std::shared_ptr<Node> node,
                              std::unordered_set<int>& subgraph_ids,
                              bool inside_subgraphs = true,
                              unsigned depth = 0);

        private:
            void build_subgraphs(std::shared_ptr<Function> func);
            NodeVector build_ck_nodes(std::shared_ptr<Function> func);

            void sanity_check(std::shared_ptr<Function> func, NodeVector& ck_nodes);
            void clean_up();

        private:
            using IDGraphMap = std::unordered_map<int, MLIRSubgraph>;
            using NodeGraphMap = std::unordered_map<std::shared_ptr<Node>, int>;
            IDGraphMap m_id_to_graph;
            NodeGraphMap m_node_to_graph;
            // Mutex over sub-graph IDs
            std::mutex m_subgraph_mutex;
            static const std::set<std::type_index> m_supported_ops;
            // Maximum depth to check for cycles during merging of sub-graphs.
            // If exceeded, we conservatively assume a cycle.
            int m_max_cycle_depth;
        };
    }
}
