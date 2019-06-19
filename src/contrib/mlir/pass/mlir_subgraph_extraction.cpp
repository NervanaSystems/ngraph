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

#include "mlir_subgraph_extraction.hpp"
#include "ngraph/node.hpp"
#include "ngraph/assertion.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/experimental/compiled_kernel.hpp"
#include "ngraph/op/get_output_element.hpp"

using namespace ngraph::descriptor;
using namespace ngraph::op;
using namespace ngraph::pass;

#define TI(x) std::type_index(typeid(x))

void MLIRSubgraphExtractionPass::MLIRSubgraph::add_inputs(NodeVector &inputs)
{
    // inputs list are not exclusive, avoid duplication
    for (NodeVector::iterator it = inputs.begin(); it != inputs.end(); it++)
    {
        std::shared_ptr<Node> node = *it;
        if (std::find(m_input_nodes.begin(), m_input_nodes.end(), node) == m_input_nodes.end())
        {
            m_input_nodes.push_back(node);
        }
    }
}

void MLIRSubgraphExtractionPass::MLIRSubgraph::add_node(std::shared_ptr<Node> node)
{
    NGRAPH_CHECK(std::find(m_nodes.begin(), m_nodes.end(), node) == m_nodes.end(), "node added to graph before");
    m_nodes.push_back(node);
}

void MLIRSubgraphExtractionPass::MLIRSubgraph::merge(MLIRSubgraph& other)
{
    // nodes  of sub-graphs are exclusive
    m_nodes.insert(m_nodes.end(), other.get_nodes().begin(), other.get_nodes().end());
    // merge inputs
    add_inputs(other.get_input_nodes());
}

// The sub-graph construction algorithm is as follows
// For each node, check its predecessors, if 
// - all predecessors in sub-graphs belong to the same sub-graph (graph ID), then extend the sub-graph to include the current node.
//   Predecessors outside sub-graphs are marked as input to the sub-graph. 
// - predecessors in sub-graphs belong to different sub-graphs, then merge all the sub-graphs into one, and add current node to it.
//   Predecessors outside sub-graphs are marked as input to the sub-graph. 
//
// For each sub-graph construction, build a CompiledKernel(CK) node around it as follows
// - all inputs edges to the sub-graph are cloned as inputs to CK node as well.
// - all outputs edges from the sub-graph are removed and added as outputs to CK node instead.
// - CK will internally have lists record graph nodes, and graph output nodes.
bool MLIRSubgraphExtractionPass::run_on_function(std::shared_ptr<Function> func)
{
    for (auto op : func->get_ordered_ops())
    {
        NodeVector inputs;
        int graph_id = -1;

        // unsupported ops, skip
        if (!is_supported_mlir_op(op))
        {
            continue;
        }
        if (TI(Parameter) == TI(*op) || TI(Result) == TI(*op))
        {
            continue;
        }

        // supported op
        for (auto pred : op->get_arguments())
        {
            int pred_graph_id = get_subgraph(pred);
            if (pred_graph_id == -1)
            {
                // predecessor doesn't belong to any sub-graph, it is an input
                inputs.push_back(pred);
            } else 
            {
                // we have a predecessor in a sub-graph
                if (graph_id == -1)
                {
                    // record first sub-graph
                    graph_id = pred_graph_id;
                }
                else
                {
                    // merge this sub-graph into first one
                    MLIRSubgraph& first_sub_graph = get_subgraph(graph_id);
                    MLIRSubgraph& pred_sub_graph = get_subgraph(pred_graph_id);
                    merge_subgraphs(first_sub_graph, pred_sub_graph);
                }
            }
        }
        if (graph_id == -1)
        {
            // we couldn't find a previous sub-graph to extend with this node
            // create a new one
            MLIRSubgraph sg;
            add_inputs_to_subgraph(sg, inputs);
            add_node_to_subgraph(sg, op);
            add_subgraph(sg);
        } 
        else
        {
            // we have a sub-graph. Update it with current node.
            MLIRSubgraph &sg = get_subgraph(graph_id);
            add_node_to_subgraph(sg, op);
            add_inputs_to_subgraph(sg, inputs);
        }
    }

    // attach CK node to each sub-graph.
     
    for (auto it : m_id_to_graph)
    {
        
        MLIRSubgraph sg = it.second;
        NodeVector outputs = std::move(get_subgraph_outputs(sg.get_nodes(), {} /*exclusions*/));
        auto ck = std::make_shared<CompiledKernel>(sg.get_nodes(), outputs, sg.get_input_nodes());
        NodeVector& nodes_list = sg.get_nodes();

        NGRAPH_DEBUG << "[CK Extract] Graph ID = " << sg.get_id() << std::endl;
        NGRAPH_DEBUG << "[CK Extract] Graph Nodes: " << std::endl;
        for (auto node : nodes_list) 
        {
            NGRAPH_DEBUG << "[CK Extract] " << *node << std::endl;
        }

        NGRAPH_DEBUG << "[CK Extract] Input Nodes: " << std::endl;
        for (auto node : sg.get_input_nodes()) 
        {
            NGRAPH_DEBUG << "[CK Extract] " << *node << std::endl;
        }

        NGRAPH_DEBUG << "[CK Extract] Output Nodes: " << std::endl;
        for (auto node : outputs) 
        {
            NGRAPH_DEBUG << "[CK Extract] " << *node << std::endl;
        }

        // Connect CompiledKernel to output nodes by replacing the output descriptors of the output
        // nodes.
        for (size_t i = 0, end = outputs.size(); i < end; ++i)
        {
            auto& output_descs = outputs[i]->get_outputs();
            NGRAPH_CHECK(output_descs.size() == 1, "Unexpected multiple output descriptors");
            auto& out_desc = output_descs[0];

            // 'replace_output' invalidates iterator of the original container. Use a copy instead.
            const std::set<descriptor::Input*> input_descs = out_desc.get_inputs();

            for (descriptor::Input* in_desc : input_descs)
            {
                in_desc->replace_output(ck, i);
            }
        }
    }

    return true;
}

#define TI(x) std::type_index(typeid(x))

bool MLIRSubgraphExtractionPass::is_supported_mlir_op(std::shared_ptr<Node> node)
{
    if (TI(Parameter) == TI(*node) || TI(Result) == TI(*node))
    {
        return true;
    }

    // supported by backend ?
    if (m_supported_ops.find(TI(*node)) == m_supported_ops.end())
    {
        return false;
    }

    // check on invariants expected by MLIR backend

    // Dot is 2D only
    if (TI(ngraph::op::Dot) == TI(*node))
    {
        if (node->get_input_shape(0).size() != 2 || node->get_input_shape(1).size() != 2)
        {
            return false;
        }
    }
    return true;
}

const std::set<std::type_index> MLIRSubgraphExtractionPass::m_supported_ops{
#define MLIR_OP(OP) TI(ngraph::op::OP),
#include "contrib/mlir/ops_supported.inc"
};
