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

void MLIRSubgraphExtractionPass::MLIRSubgraph::add_outputs(NodeVector &outputs)
{
    m_output_nodes.insert(m_output_nodes.end(), outputs.begin(), outputs.end());
}

void MLIRSubgraphExtractionPass::MLIRSubgraph::add_node(std::shared_ptr<Node> node)
{
    NGRAPH_CHECK(std::find(m_nodes.begin(), m_nodes.end(), node) == m_nodes.end(), "node added to graph before");
    m_nodes.push_back(node);
}

void MLIRSubgraphExtractionPass::MLIRSubgraph::merge(MLIRSubgraph& other)
{
    NGRAPH_CHECK (&other != this, "Cannot merge a sub-graph into itself");

    // nodes  of sub-graphs are exclusive
    m_nodes.insert(m_nodes.end(), other.get_nodes().begin(), other.get_nodes().end());
    // merge inputs
    add_inputs(other.get_inputs());
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
        int first_graph_id = -1;
        std::vector<int> subgraph_ids;
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
            int pred_subgraph_id = get_subgraph(pred);
            if (pred_subgraph_id == -1)
            {
                // predecessor doesn't belong to any sub-graph, it is an input
                inputs.push_back(pred);
            } 
            else 
            {
                // record sub-graph id of the predecessor
                subgraph_ids.push_back(pred_subgraph_id);
            }
        }
        if (subgraph_ids.size() == 0)
        {
            // we couldn't find any predecessor sub-graphs to extend with this node
            // create a new sub-graph
            MLIRSubgraph sg;
            add_inputs_to_subgraph(sg, inputs);
            add_node_to_subgraph(sg, op);
            add_subgraph(sg);
        } 
        else
        {
            // we have sub-graphs. 
            // check if adding this node to the sub-graph will create a cycle in the DAG
            if (!check_cycles(inputs, subgraph_ids))
            {
                // merge sub-graphs if needed
                MLIRSubgraph& first_subgraph = get_subgraph(subgraph_ids[0]);
                for (auto i = 1; i < subgraph_ids.size(); i++)
                {
                    MLIRSubgraph& subgraph = get_subgraph(subgraph_ids[i]);
                    merge_subgraphs(first_subgraph, subgraph);
                }

                add_node_to_subgraph(first_subgraph, op);
                add_inputs_to_subgraph(first_subgraph, inputs);
            }
            else
            {
                // we have a cycle, start a new sub-graph
                MLIRSubgraph sg;
                // use all predecessors as graph inputs 
                NodeVector inputs = op->get_arguments();
                add_inputs_to_subgraph(sg, inputs);
                add_node_to_subgraph(sg, op);
                add_subgraph(sg);
            }
        }
    }

    // get output nodes for each sub-graph. Do this before attachin CK nodes since we will 
    // remove output edges from the sub-graphs. 
    for (IDGraphMap::iterator it = m_id_to_graph.begin(); it != m_id_to_graph.end(); it++)
    {
        MLIRSubgraph& sg = it->second;
        NodeVector outputs = std::move(get_subgraph_outputs(sg.get_nodes(), {} /*exclusions*/, false /* ignore unused */, false /* ignore output duplicates */));
        sg.add_outputs(outputs);
    }
    // attach CK node to each sub-graph.
     
    for (auto it : m_id_to_graph)
    {
        
        MLIRSubgraph sg = it.second;
        auto outputs = sg.get_outputs();
        auto ck = std::make_shared<CompiledKernel>(sg.get_nodes(), outputs, sg.get_inputs());
        NodeVector& nodes_list = sg.get_nodes();

        NGRAPH_DEBUG << "[CK Extract] Graph ID = " << sg.get_id() << std::endl;
        NGRAPH_DEBUG << "[CK Extract] Graph Nodes: " << std::endl;
        for (auto node : nodes_list) 
        {
            NGRAPH_DEBUG << "[CK Extract] " << *node << std::endl;
        }

        NGRAPH_DEBUG << "[CK Extract] Input Nodes: " << std::endl;
        for (auto node : sg.get_inputs()) 
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

bool MLIRSubgraphExtractionPass::check_cycles(NodeVector& inputs, std::vector<int>& subgraph_ids)
{
    NodeVector work_list;
    work_list.insert(work_list.end(), inputs.begin(), inputs.end());
    while(!work_list.empty())
    {
        auto node = work_list.back();
        work_list.pop_back();
        if (std::find(subgraph_ids.begin(), subgraph_ids.end(), get_subgraph(node)) != subgraph_ids.end())
        {
            // we hit one of the sub-graphs we want to extend. we have a cycle.
            NGRAPH_DEBUG << "[CK Extract] Cycle found when trying to add node" << std::endl;
            return true;
        }
        for (auto pred : node->get_arguments())
        {
            work_list.push_back(pred);
        }
    }
    return false;
}

const std::set<std::type_index> MLIRSubgraphExtractionPass::m_supported_ops{
#define MLIR_OP(OP) TI(ngraph::op::OP),
#include "contrib/mlir/ops_supported.inc"
};
