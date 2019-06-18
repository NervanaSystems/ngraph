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

void MLIRSubgraphExtractionPass::MLIRSubgraph::add_node(std::shared_ptr<Node> node, bool is_head)
{
    NGRAPH_CHECK(std::find(m_nodes.begin(), m_nodes.end(), node) == m_nodes.end(), "node added to graph before");
    m_nodes.push_back(node);
    if (is_head) 
    {
        NGRAPH_CHECK(std::find(m_head_nodes.begin(), m_head_nodes.end(), node) == m_head_nodes.end(), "head node added to graph before");
        m_head_nodes.push_back(node);
    }

}

void MLIRSubgraphExtractionPass::MLIRSubgraph::merge(MLIRSubgraph& other)
{
    // nodes and head nodes of sub-graphs are exclusive
    m_nodes.insert(m_nodes.end(), other.get_nodes().begin(), other.get_nodes().end());
    m_head_nodes.insert(m_head_nodes.end(), other.get_head_nodes().begin(), other.get_head_nodes().end());
    // merge inputs
    add_inputs(other.get_input_nodes());
}

bool MLIRSubgraphExtractionPass::run_on_function(std::shared_ptr<Function> func)
{
    for (auto op : func->get_ordered_ops())
    {
        NodeVector inputs;
        int graph_id = -1;
        bool is_head = false;
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
        // go over all its inputs and figure out which graph ID it belongs to
        // if all inputs are outside any sub-graph, we assign a new graph ID to that node
        // if some inputs belong to different sub-graphs, we merge those sub-graphs into the first one
        for (auto pred : op->get_arguments())
        {
            int pred_graph_id = get_subgraph(pred);
            if (pred_graph_id == -1)
            {
                // predecessor doesn't belong to any sub-graph, it is an input
                inputs.push_back(pred);
                // this makes the current node a head node (taking input from outside the graph)
                is_head = true;
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
            add_node_to_subgraph(sg, op, is_head);
            add_subgraph(sg);
        } 
        else
        {
            // we have a sub-graph. Update it with current node.
            MLIRSubgraph &sg = get_subgraph(graph_id);
            add_node_to_subgraph(sg, op, is_head);
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
        // remove all input edges to the sub-graph
        for (auto head : sg.get_head_nodes())
        {
            for (descriptor::Input& input : head->get_inputs())
            {
                if (sg.is_input_node(input.get_output().get_node()))
                {
                    ck->add_head_node_input(head.get(), &input.get_tensor());
                    input.remove_output();
                }
            }
        }
        #if 0
// remove input edges to sub-graph
        for (auto arg : sg.get_input_nodes())
        {
            // Find edges from input nodes that go into the sub-graph. Replace them with CK output.
            for (auto output : arg->outputs())
            {
                // make a copy since modifying the inputs list will corrupt the container iterator
                auto inputs = output.get_target_inputs();
                // all inputs that this output feeds
                for (auto use : inputs)
                {
                    if (std::find(nodes_list.begin(), nodes_list.end(), use.get_node()->shared_from_this()) != nodes_list.end())
                    {
                        // find uses inside the sub-graph. Replace source with corresponding output of CK
                        output.remove_target_input(use);
                    }
                }
            }
        }
        


        // Replace input edges to sub-graph with output of CK instead. 
        // This ensures the sub-graph is unreachable from the rest of the graph
        unsigned i = 0;
        for (auto arg : sg.get_input_nodes())
        {
            // Find edges from input nodes that go into the sub-graph. Replace them with CK output.
            for (auto output : arg->outputs())
            {
                // make a copy since modifying the inputs list will corrupt the container iterator
                auto inputs = output.get_target_inputs();
                // all inputs that this output feeds
                for (auto use : inputs)
                {
                    if (std::find(nodes_list.begin(), nodes_list.end(), use.get_node()->shared_from_this()) != nodes_list.end())
                    {
                        // find uses inside the sub-graph. Replace source with corresponding output of CK
                        use.replace_source_output(ck->output(i));
                    }
                }
            }
            i++;
        }
        #endif
    }
#if 0
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

        #endif
 
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
