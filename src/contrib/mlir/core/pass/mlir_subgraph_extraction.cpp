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

#include "mlir_subgraph_extraction.hpp"
#include "ngraph/assertion.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "ngraph/ops.hpp"

using namespace ngraph::descriptor;
using namespace ngraph::op;
using namespace ngraph::pass;

int MLIRSubgraphExtractionPass::MLIRSubgraph::m_curr_graph_id = 0;

void MLIRSubgraphExtractionPass::MLIRSubgraph::add_inputs(NodeVector& inputs)
{
    // inputs list are not exclusive, avoid duplication
    for (auto node : inputs)
    {
        if (m_input_node_set.insert(node).second)
        {
            m_input_node_vector.push_back(node);
        }
    }
}

void MLIRSubgraphExtractionPass::MLIRSubgraph::add_outputs(NodeVector& outputs)
{
    m_output_nodes.insert(m_output_nodes.end(), outputs.begin(), outputs.end());
}

void MLIRSubgraphExtractionPass::MLIRSubgraph::add_node(std::shared_ptr<Node> node)
{
    NGRAPH_CHECK(m_pass.m_node_to_graph.find(node) == m_pass.m_node_to_graph.end(),
                 "node added to graph before");
    m_nodes.emplace_back(node);
    m_pass.m_node_to_graph[node] = get_id();
}

// The sub-graph construction algorithm is as follows
// Construct a map of node to number of its input not being processes
// Put the node with value 0 into a ready list
// Go through the nodes in the ready list until the list is empty:
// - if the last node processed is supported, try to find a supported node and add that node to the
//   current sub-graph.
// - if none is available, process an unsupported node.
// - if the last node processed is unsupported, try to find an unsupported node.
// - if none is available, start a new sub-graph, find a supported node and add that node to the new
//   sub-graph.
// - Erase processed node form the ready list, update the value of its successors in the map, and
//   add its successor to ready list if value is 0.
//
// Sub-graph may contain multiple disjoint clusters.
//
// For each sub-graph found build a CompiledKernel(CK) node around it as follows
// - all inputs edges to the sub-graph are cloned as inputs to CK node as well.
// - all outputs edges from the sub-graph are removed and added as outputs to CK node instead.
// - CK will internally have lists record graph nodes, and graph output nodes.
bool MLIRSubgraphExtractionPass::run_on_function(std::shared_ptr<Function> func)
{
    build_subgraphs(func);
    auto ck_nodes = build_ck_nodes(func);

#ifdef NGRAPH_DEBUG_ENABLE
    sanity_check(func, ck_nodes);
#endif

    clean_up();

    return true;
}

static void
    process_successors(std::shared_ptr<ngraph::Node> node,
                       std::unordered_map<std::shared_ptr<ngraph::Node>, size_t>& node_to_size_map,
                       std::list<std::shared_ptr<ngraph::Node>>& nodes_ready)
{
    for (auto output : node->outputs())
    {
        for (auto input : output.get_target_inputs())
        {
            auto user = input.get_node()->shared_from_this();
            node_to_size_map[user]--;
            if (node_to_size_map[user] == 0)
            {
                nodes_ready.push_back(user);
            }
        }
    }
}

void MLIRSubgraphExtractionPass::process_supported_op(std::shared_ptr<ngraph::Node> node,
                                                      int current_subgraph_id)
{
    NodeVector inputs;
    for (auto pred : node->get_arguments())
    {
        int pred_subgraph_id = get_subgraph_id(pred);
        if (pred_subgraph_id != current_subgraph_id)
        {
            // predecessor doesn't belong to current sub-graph, it is an
            // input
            inputs.push_back(pred);
        }
    }
    // add inputs and op to current sub-graph
    MLIRSubgraph& current_subgraph = get_subgraph(current_subgraph_id);
    current_subgraph.add_node(node);
    current_subgraph.add_inputs(inputs);
    NGRAPH_DEBUG << "[CK Extract] Node Processed " << *node;
}

static void erase_node(std::list<std::shared_ptr<ngraph::Node>>::iterator& it,
                       std::list<std::shared_ptr<ngraph::Node>>& nodes_ready)
{
    auto old_it = it;
    it++;
    nodes_ready.erase(old_it);
}

void MLIRSubgraphExtractionPass::build_subgraphs(std::shared_ptr<Function> func)
{
    NGRAPH_DEBUG << "[CK Extract] Construct sub-graphs";
    int current_subgraph_id = 0;

    std::unordered_map<std::shared_ptr<Node>, size_t> node_to_size_map;
    std::list<std::shared_ptr<Node>> nodes_ready;
    bool last_op_is_supported = false;

    for (auto op : func->get_ops())
    {
        size_t arg_count = op->get_input_size();
        node_to_size_map[op] = arg_count;
        if (arg_count == 0)
        {
            nodes_ready.push_back(op);
        }
    }

    bool change_mode = false;
    while (!nodes_ready.empty())
    {
        for (auto it = nodes_ready.begin(); it != nodes_ready.end();)
        {
            auto node = *it;
            if (is_type<Result>(node))
            {
                erase_node(it, nodes_ready);
            }
            else if (is_type<Parameter>(node))
            {
                process_successors(node, node_to_size_map, nodes_ready);
                erase_node(it, nodes_ready);
            }
            else if (is_supported_mlir_op(node))
            {
                if (last_op_is_supported)
                {
                    process_supported_op(node, current_subgraph_id);
                    process_successors(node, node_to_size_map, nodes_ready);
                    erase_node(it, nodes_ready);
                    change_mode = false;
                }
                else
                {
                    change_mode = true;
                    it++;
                }
            }
            else
            {
                if (last_op_is_supported)
                {
                    change_mode = true;
                    it++;
                }
                else
                {
                    process_successors(node, node_to_size_map, nodes_ready);
                    erase_node(it, nodes_ready);
                    change_mode = false;
                }
            }
        }

        if (change_mode)
        {
            if (last_op_is_supported)
            {
                for (auto it = nodes_ready.begin(); it != nodes_ready.end();)
                {
                    auto node = *it;
                    if (is_type<Result>(node))
                    {
                        erase_node(it, nodes_ready);
                    }
                    else if (!is_supported_mlir_op(node))
                    {
                        process_successors(node, node_to_size_map, nodes_ready);
                        erase_node(it, nodes_ready);
                        change_mode = false;
                        last_op_is_supported = false;
                    }
                    else
                    {
                        it++;
                    }
                }
            }
            else
            {
                // create a new sub-graph
                MLIRSubgraph sg = MLIRSubgraph::create(this);
                NGRAPH_DEBUG << "   [CK Extract] Start new sub-graph " << sg.get_id();
                add_subgraph(sg);
                current_subgraph_id = sg.get_id();
                for (auto it = nodes_ready.begin(); it != nodes_ready.end();)
                {
                    auto node = *it;
                    if (is_type<Result>(node))
                    {
                        erase_node(it, nodes_ready);
                    }
                    else if (is_supported_mlir_op(node))
                    {
                        process_supported_op(node, current_subgraph_id);
                        process_successors(node, node_to_size_map, nodes_ready);
                        erase_node(it, nodes_ready);
                        change_mode = false;
                        last_op_is_supported = true;
                    }
                    else
                    {
                        it++;
                    }
                }
            }
        }
    }

    NGRAPH_DEBUG << "[CK Extract] Get subgraphs output nodes";
    // get output nodes for each sub-graph. Do this before attaching CK nodes since we will
    // remove output edges from the sub-graphs.
    for (IDGraphMap::iterator it = m_id_to_graph.begin(); it != m_id_to_graph.end(); it++)
    {
        MLIRSubgraph& sg = it->second;
        auto& nodes = sg.get_nodes();
        NodeVector outputs = get_subgraph_outputs(NodeVector(nodes.begin(), nodes.end()),
                                                  {} /*exclusions*/,
                                                  false /* ignore unused */,
                                                  false /* ignore output duplicates */);
        sg.add_outputs(outputs);
    }
}

ngraph::NodeVector MLIRSubgraphExtractionPass::build_ck_nodes(std::shared_ptr<Function> func)
{
    NodeVector ck_nodes;
    NGRAPH_DEBUG << "[CK Extract] Construct CK nodes";
    // attach CK node to each sub-graph.
    for (auto it : m_id_to_graph)
    {
        MLIRSubgraph sg = it.second;
        auto& inputs = sg.get_inputs();
        auto& outputs = sg.get_outputs();
        auto& nodes = sg.get_nodes();

        NodeVector inputs_vector(inputs.begin(), inputs.end());
        NodeVector outputs_vector(outputs.begin(), outputs.end());
        // must store nodes in topological order
        auto nodes_list = subgraph_topological_sort(nodes);
        NodeVector nodes_vector(nodes_list.begin(), nodes_list.end());
        auto ck = std::make_shared<CompiledKernel>(nodes_vector, outputs_vector, inputs_vector);

        ck_nodes.push_back(ck);

        NGRAPH_DEBUG << "[CK Extract] Graph ID = " << sg.get_id();
        NGRAPH_DEBUG << "   [CK Extract] Graph Nodes: ";
        for (auto node : nodes)
        {
            NGRAPH_DEBUG << "   [CK Extract] " << *node;
        }

        NGRAPH_DEBUG << "   [CK Extract] Input Nodes: ";
        for (auto node : inputs)
        {
            NGRAPH_DEBUG << "   [CK Extract] " << *node;
        }

        NGRAPH_DEBUG << "   [CK Extract] Output Nodes: ";
        for (auto node : outputs)
        {
            NGRAPH_DEBUG << "   [CK Extract] " << *node;
            ;
        }
        NGRAPH_DEBUG << "   [CK Extract] CK Node = " << *ck;
    }

    // Connect CompiledKernel to output nodes by replacing the output descriptors of the output
    // Do this after all CK nodes are constructed since they add new edges in the graph (CK inputs)
    for (auto& node : ck_nodes)
    {
        auto ck = std::static_pointer_cast<CompiledKernel>(node);

        auto& outputs_vector = ck->get_kernel_outputs();
        auto& node_list = ck->get_node_list();
        std::unordered_set<std::shared_ptr<Node>> node_set(node_list.begin(), node_list.end());

        for (size_t i = 0, end = outputs_vector.size(); i < end; ++i)
        {
            auto& output_descs = outputs_vector[i]->get_outputs();
            NGRAPH_CHECK(output_descs.size() == 1, "Unexpected multiple output descriptors");
            auto& out_desc = output_descs[0];

            // 'replace_output' invalidates iterator of the original container. Use a copy instead.
            const std::vector<descriptor::Input*> input_descs = out_desc.get_inputs();

            for (descriptor::Input* in_desc : input_descs)
            {
                if (node_set.find(in_desc->get_node()) == node_set.end())
                {
                    in_desc->replace_output(ck, i);
                }
            }
        }
    }
    for (auto& node : ck_nodes)
    {
        auto ck = std::static_pointer_cast<CompiledKernel>(node);
        if (ck->get_output_size() > 1)
        {
            for (auto& old_output : ck->outputs())
            {
                auto inputs = old_output.get_target_inputs();
                auto goe_node = old_output.as_single_output_node(false);
                auto new_output = goe_node->output(0);
                for (auto& input : inputs)
                {
                    input.replace_source_output(new_output);
                }
            }
        }
    }

    return ck_nodes;
}

// Do a sanity check on graph invariants
//  - no cycles
//  - inputs to sub-graph are inputs to CK
//  - no outputs out of subgraph for output nodes
void MLIRSubgraphExtractionPass::sanity_check(std::shared_ptr<Function> func, NodeVector& ck_nodes)
{
    NodeVector cycles;
    bool is_bkwd_cycle;
    if (check_for_cycles(func.get(), cycles, is_bkwd_cycle))
    {
        NGRAPH_CHECK(cycles.size() != 0, "Empty cycle ?");
        if (is_bkwd_cycle)
        {
            NGRAPH_DEBUG << "Backward cycle:";
        }
        for (auto& node : cycles)
        {
            NGRAPH_DEBUG << node;
        }

        NGRAPH_UNREACHABLE("Function contains cycle after subgraph constructions");
    }

    for (auto& node : ck_nodes)
    {
        auto ck_node = std::static_pointer_cast<CompiledKernel>(node);
        auto& node_list = ck_node->get_node_list();
        std::unordered_set<std::shared_ptr<Node>> node_set(node_list.begin(), node_list.end());
        // CK output nodes shouldn't have any users outside the sub-graph,
        // they are all moved to the CK node instead
        for (auto& ck_output : ck_node->get_kernel_outputs())
        {
            for (auto& user : ck_output->get_users())
            {
                NGRAPH_CHECK(node_set.find(user) != node_set.end(),
                             "CK output nodes users should be in the sub-graph");
            }
        }

        // Any input to CK must not have any user in the sub-graph body
        for (auto& arg : ck_node->get_arguments())
        {
            bool found = false;
            for (auto& user : arg->get_users())
            {
                found = (node_set.find(user) == node_set.end());
                if (found)
                {
                    break;
                }
            }
            NGRAPH_CHECK(found, "CK input is input to sub-graph");
        }
    }
}

bool MLIRSubgraphExtractionPass::is_supported_mlir_op(std::shared_ptr<Node> node)
{
    if (is_type<Parameter>(node) || is_type<Result>(node))
    {
        return true;
    }

    // supported by backend ?
    auto& supportedOps = getSupportedOps();
    if (supportedOps.find(node->get_type_info()) == supportedOps.end())
    {
        return false;
    }

    // check on invariants expected by MLIR backend

    if (auto div = as_type_ptr<ngraph::op::Divide>(node))
    {
        if (div->is_pythondiv())
        {
            // Python specific division rounding is not supported yet.
            return false;
        }

        return true;
    }

    // Dot is 2D only
    if (is_type<ngraph::op::Dot>(node))
    {
        if (node->get_input_shape(0).size() != 2 || node->get_input_shape(1).size() != 2)
        {
            return false;
        }
        else
        {
            return true;
        }
    }

    if (auto conv_node = as_type_ptr<ngraph::op::Convolution>(node))
    {
        // No padding for now
        auto pad_below = conv_node->get_padding_below();
        auto pad_above = conv_node->get_padding_above();
        auto data_dilation = conv_node->get_data_dilation_strides();
        auto window_dilation = conv_node->get_window_dilation_strides();

        auto is_one = [](size_t s) { return s == 1; };

        return std::all_of(data_dilation.begin(), data_dilation.end(), is_one) &&
               std::all_of(window_dilation.begin(), window_dilation.end(), is_one);
    }

    // MKLDNN only supports softmax across single axis
    if (auto softmax = as_type_ptr<ngraph::op::Softmax>(node))
    {
        // Softmax is only supported through callback
        if (std::getenv("NGRAPH_MLIR_CALLBACK") == nullptr)
        {
            return false;
        }
        auto arg0_shape = node->get_input_shape(0);
        auto arg0_rank = arg0_shape.size();

        return (arg0_rank == 4 || arg0_rank == 2) &&
               node->get_input_element_type(0) == element::f32 && softmax->get_axes().size() == 1;
    }

    if (auto avg_pool = as_type_ptr<ngraph::op::AvgPool>(node))
    {
        // AvgPool is only supported through callback
        if (std::getenv("NGRAPH_MLIR_CALLBACK") == nullptr)
        {
            return false;
        }
        auto arg0_shape = node->get_input_shape(0);
        auto arg0_rank = arg0_shape.size();

        return ((arg0_rank == 4 && avg_pool->get_window_shape().size() == 2) ||
                (arg0_rank == 5 && avg_pool->get_window_shape().size() == 3)) &&
               node->get_input_element_type(0) == element::f32;
    }

    if (auto avg_pool_backprop = as_type_ptr<ngraph::op::AvgPoolBackprop>(node))
    {
        // AvgPoolBackprop is only supported through callback
        if (std::getenv("NGRAPH_MLIR_CALLBACK") == nullptr)
        {
            return false;
        }
        auto arg0_shape = node->get_input_shape(0);
        auto arg0_rank = arg0_shape.size();

        return ((arg0_rank == 4 && avg_pool_backprop->get_window_shape().size() == 2) ||
                (arg0_rank == 5 && avg_pool_backprop->get_window_shape().size() == 3)) &&
               node->get_input_element_type(0) == element::f32;
    }

    if (auto max_pool_backprop = as_type_ptr<ngraph::op::MaxPoolBackprop>(node))
    {
        // MaxPoolBackprop is only supported through callback
        if (std::getenv("NGRAPH_MLIR_CALLBACK") == nullptr)
        {
            return false;
        }
        auto arg0_shape = node->get_input_shape(0);
        auto arg0_rank = arg0_shape.size();

        return ((arg0_rank == 4 && max_pool_backprop->get_window_shape().size() == 2) ||
                (arg0_rank == 5 && max_pool_backprop->get_window_shape().size() == 3)) &&
               node->get_input_element_type(0) == element::f32;
    }

    if (auto max_pool = as_type_ptr<ngraph::op::MaxPool>(node))
    {
        // MaxPool is only supported through callback
        if (std::getenv("NGRAPH_MLIR_CALLBACK") == nullptr)
        {
            return false;
        }
        auto arg0_shape = node->get_input_shape(0);
        auto arg0_rank = arg0_shape.size();

        return ((arg0_rank == 4 && max_pool->get_window_shape().size() == 2) ||
                (arg0_rank == 5 && max_pool->get_window_shape().size() == 3)) &&
               node->get_input_element_type(0) == element::f32;
    }

    if (is_type<ngraph::op::MatMul>(node))
    {
        // MatMul is only supported through callback
        if (std::getenv("NGRAPH_MLIR_CALLBACK") == nullptr)
        {
            return false;
        }
    }

    if (is_type<ngraph::op::Gemm>(node))
    {
        // Gemm is only supported through callback
        if (std::getenv("NGRAPH_MLIR_CALLBACK") == nullptr)
        {
            return false;
        }
    }

    return true;
}

void MLIRSubgraphExtractionPass::clean_up()
{
    m_id_to_graph.clear();
    m_node_to_graph.clear();
}

const std::set<ngraph::Node::type_info_t>& MLIRSubgraphExtractionPass::getSupportedOps()
{
    static std::set<Node::type_info_t> supportedOps{
#define MLIR_OP(OP) OP::type_info,
#include "contrib/mlir/core/ops_supported.inc"
    };
    return supportedOps;
}
