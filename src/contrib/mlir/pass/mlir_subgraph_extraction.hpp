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
                /// Create a sub-graph with a specific ID.
                MLIRSubgraph(int graph_id) : m_graph_id(graph_id) {}
                /// Get graph id
                int get_id() const      { return m_graph_id; }
                /// Get all nodes in the sub-graph. Including head nodes. 
                NodeVector& get_nodes() { return m_nodes; }
                /// Get input nodes. Predecessors to head nodes. 
                NodeVector& get_input_nodes() { return m_input_nodes; }
                /// Get head nodes. Nodes in the sub-graph that has at least one input from outside.
                NodeVector& get_head_nodes()  { return m_head_nodes;  }
                /// Checks if a node is an input node.
                bool is_input_node(std::shared_ptr<Node> node)
                {
                    return std::find(m_input_nodes.begin(), m_input_nodes.end(), node) != m_input_nodes.end();
                }
                /// Add a list of input nodes to the graph.
                void add_inputs(NodeVector &inputs);
                /// Add one node to the sub-graph. If is_head is true, it is added to head nodes as well.
                void add_node(std::shared_ptr<Node> node, bool is_head = false);
                /// Merges sub-graph (other) into this sub-graph.
                void merge(MLIRSubgraph& other);

            private:
                // Unique ID for this sub-graph. 
                int m_graph_id;
                // Actual nodes of the sub-graph
                NodeVector m_nodes;
                // Predecessor to head nodes in the sub-graph. 
                NodeVector m_input_nodes;
                // Head nodes in the sub-graph (taking at least one input from outside the sub-graph)
                NodeVector m_head_nodes;
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
            void add_subgraph(MLIRSubgraph sg)
            {
                m_id_to_graph.emplace(sg.get_id(), sg);
            }
            void add_node_to_subgraph(MLIRSubgraph& sg, std::shared_ptr<Node> node, bool is_head = false)
            {
                sg.add_node(node, is_head);
                m_node_to_graph[node] = sg.get_id();
            }
            void add_inputs_to_subgraph(MLIRSubgraph& sg, NodeVector& inputs)
            {
                sg.add_inputs(inputs);
            }
            void merge_subgraphs(MLIRSubgraph& sg1, MLIRSubgraph& sg2)
            {
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
