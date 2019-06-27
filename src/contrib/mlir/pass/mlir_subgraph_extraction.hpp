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
        /// This pass creates CompiledKernel ops enclosing maximal sub-graphs of ops that are supported by MLIR
        class MLIRSubgraphExtractionPass : public ngraph::pass::FunctionPass
        {
        private:
            class MLIRSubgraph
            {
            private:
                static int get_new_graph_id()
                {
                    static int graph_id = 0;
                    return graph_id++;
                }

            public:
                /// Create a sub-graph with a new ID.
                MLIRSubgraph() 
                {
                    m_graph_id = MLIRSubgraph::get_new_graph_id();
                }
                /// Get graph id
                int get_id() const      { return m_graph_id; }
                /// Get all nodes in the sub-graph.
                NodeVector& get_nodes() { return m_nodes; }
                /// Get input nodes. Predecessors to head nodes. 
                NodeVector& get_inputs() { return m_input_nodes; }
                NodeVector& get_outputs() { return m_output_nodes; }
                /// Add a list of input nodes to the graph.
                void add_inputs(NodeVector &inputs);
                /// Add one node to the sub-graph.
                void add_node(std::shared_ptr<Node> node);
                /// Merges sub-graph (other) into this sub-graph.
                void merge(MLIRSubgraph& other);
                void add_outputs(NodeVector &outputs);

            private:
                // Unique ID for this sub-graph. 
                int m_graph_id;
                // Actual nodes of the sub-graph
                NodeVector m_nodes;
                // Predecessor to head nodes in the sub-graph. 
                NodeVector m_input_nodes;
                NodeVector m_output_nodes;
            };

        public:
            MLIRSubgraphExtractionPass() {}
            bool run_on_function(std::shared_ptr<Function> func) override;
            /// Checks if an ngraph node is supported by MLIR backend
            bool is_supported_mlir_op(std::shared_ptr<Node> node);
            /// Get the sub-graph a node belongs to
            int get_subgraph(std::shared_ptr<Node> node)
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
            void add_subgraph(MLIRSubgraph sg)
            {
                m_id_to_graph.emplace(sg.get_id(), sg);
            }
            /// Adds a node to the sub-graph
            void add_node_to_subgraph(MLIRSubgraph& sg, std::shared_ptr<Node> node)
            {
                sg.add_node(node);
                m_node_to_graph[node] = sg.get_id();
            }
            /// Adds a list of input nodes to a sub-graph
            void add_inputs_to_subgraph(MLIRSubgraph& sg, NodeVector& inputs)
            {
                sg.add_inputs(inputs);
            }
            /// Merge two sub-graphs and update maps accordingly
            void merge_subgraphs(MLIRSubgraph& sg1, MLIRSubgraph& sg2)
            {
                NGRAPH_CHECK(&sg1 != &sg2, "Cannot merge a graph into itself");
                sg1.merge(sg2);
                // Remove sub-graph from map
                m_id_to_graph.erase(sg2.get_id());
                // Associate nodes of second sub-graph to first one
                auto sg_nodes = sg2.get_nodes();
                for (auto node : sg_nodes)
                {
                    m_node_to_graph[node] = sg1.get_id();
                }
            }

            /// Checks if adding a node to an extracted sub-graph will cause a DAG cycle
            /// inputs: are the list of input nodes outside sub-graphs to the node we want to add.
            /// subgraph_ids: are the sub-graphs the predecessor the node belong to.
            /// It traverses backwards from all input nodes and checks if we reach any node that already 
            /// belongs to one of the sub-graph ids. If so, we have a cycle. 
            ///
            /// Example:
            /// A(1)
            /// |   \
            /// B(1) C
            /// |  /
            /// D
            /// we want to add D to sub-graph 1. C is an input to D. sugraph_ids are 1
            /// we traverse backwards C->A(1) and find 1, then we cannot add D since we will form a cycle
            bool check_cycles(NodeVector& inputs, std::vector<int>& subgraph_ids);

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
