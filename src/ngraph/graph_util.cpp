/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <cassert>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "ngraph/node_vector.hpp"
#include "ngraph/ops/constant.hpp"
#include "ngraph/ops/parameter.hpp"
#include "ngraph/ops/result.hpp"
#include "ngraph/ops/result_vector.hpp"
#include "ngraph/placement.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

void ngraph::traverse_nodes(const std::shared_ptr<const Function> p,
                            std::function<void(std::shared_ptr<Node>)> f)
{
    traverse_nodes(p.get(), f);
}

void ngraph::traverse_nodes(const Function* p, std::function<void(std::shared_ptr<Node>)> f)
{
    std::unordered_set<std::shared_ptr<Node>> instances_seen;
    std::deque<std::shared_ptr<Node>> stack;

    for (auto r : p->get_results())
    {
        stack.push_front(r);
    }

    for (auto param : p->get_parameters())
    {
        stack.push_front(param);
    }

    while (stack.size() > 0)
    {
        std::shared_ptr<Node> n = stack.front();
        if (instances_seen.count(n) == 0)
        {
            instances_seen.insert(n);
            f(n);
        }
        stack.pop_front();
        for (auto arg : n->get_input_ops())
        {
            if (instances_seen.count(arg) == 0)
            {
                stack.push_front(arg);
            }
        }
    }
}

void ngraph::traverse_functions(std::shared_ptr<ngraph::Function> p,
                                std::function<void(shared_ptr<Function>)> f)
{
    std::unordered_set<shared_ptr<Function>> instances_seen;
    deque<shared_ptr<Function>> stack;

    stack.push_front(p);

    while (stack.size() > 0)
    {
        shared_ptr<Function> func = stack.front();
        if (instances_seen.find(func) == instances_seen.end())
        {
            instances_seen.insert(func);
            f(func);
        }
        stack.pop_front();
        for (shared_ptr<Node> op : func->get_ops())
        {
            for (shared_ptr<Function> fp : op->get_functions())
            {
                stack.push_front(fp);
            }
        }
    }
}

void ngraph::free_nodes(shared_ptr<Function> p)
{
    std::deque<Node*> sorted_list;

    traverse_nodes(p, [&](shared_ptr<Node> n) { sorted_list.push_front(n.get()); });

    for (Node* n : sorted_list)
    {
        n->clear_arguments();
    }
}

void ngraph::replace_node(std::shared_ptr<Node> target, std::shared_ptr<Node> replacement)
{
    if (target->is_output())
    {
        throw ngraph_error("Result nodes cannot be replaced.");
    }

    // Fix input/output descriptors
    assert(target->get_outputs().size() == replacement->get_outputs().size());

    // For each of target's output O with replacement output O_rep:
    //     For each O's connected downstream input I:
    //         Change I's connected upstream output to O_rep
    for (size_t i = 0; i < target->get_outputs().size(); i++)
    {
        auto& target_output = target->get_outputs().at(i);
        std::set<ngraph::descriptor::Input*> copy_inputs{begin(target_output.get_inputs()),
                                                         end(target_output.get_inputs())};
        for (auto input : copy_inputs)
        {
            input->replace_output(replacement->get_outputs().at(i));
        }
    }

    // Fix users and arguments
    replace_node_users_arguments(target, replacement);
}

void ngraph::replace_node_users_arguments(std::shared_ptr<Node> target,
                                          std::shared_ptr<Node> replacement)
{
    for (auto user : target->users())
    {
        auto& args = const_cast<ngraph::NodeVector&>(user->get_arguments_FOR_GRAPH_REWRITE_ONLY());
        auto it = std::find(begin(args), end(args), target);
        assert(it != end(args));
        it = args.erase(it);
        args.insert(it, replacement);
        const_cast<std::multiset<Node*>&>(replacement->users()).insert(user);
    }
    const_cast<std::multiset<Node*>&>(target->users()).clear();
}

std::list<std::shared_ptr<ngraph::Node>>
    ngraph::topological_sort(const std::list<std::shared_ptr<Node>>& nodes)
{
    deque<ngraph::Node*> independent_nodes;
    unordered_map<const ngraph::Node*, size_t> node_depencency_count;
    unordered_map<ngraph::Node*, shared_ptr<ngraph::Node>> node_map;

    for (auto node : nodes)
    {
        node_map[node.get()] = node;
        node_depencency_count[node.get()] = node->get_input_ops().size();
        if (node->get_input_ops().size() == 0)
        {
            independent_nodes.push_back(node.get());
        }
    }

    list<shared_ptr<ngraph::Node>> result_list;
    while (independent_nodes.size() > 0)
    {
        auto independent_node = independent_nodes.front();
        result_list.push_back(node_map[independent_node]);
        independent_nodes.pop_front();

        for (auto user : independent_node->users())
        {
            node_depencency_count[user] -= 1;
            size_t count = node_depencency_count[user];
            if (count == 0)
            {
                independent_nodes.push_back(user);
            }
        }
    }

    return result_list;
}

void ngraph::NodeMap::update(std::shared_ptr<ngraph::Node> orig, std::shared_ptr<ngraph::Node> val)
{
    if (!exists(orig))
    {
        throw ngraph_error("Node doesn't exist!");
    }
    m_node_map[orig] = val;
}

void ngraph::NodeMap::add(std::shared_ptr<ngraph::Node> orig,
                          std::shared_ptr<ngraph::Node> replacement)
{
    if (exists(orig))
    {
        throw ngraph_error("NodeMap: key already exists");
    }
    m_node_map[orig] = replacement;
}

std::shared_ptr<ngraph::Node> ngraph::NodeMap::get(std::shared_ptr<ngraph::Node> orig) const
{
    if (!exists(orig))
    {
        throw ngraph_error("NodeMap: key does not exist");
    }
    return m_node_map.at(orig);
}

std::list<std::shared_ptr<ngraph::Node>>
    ngraph::clone_nodes(const std::list<std::shared_ptr<ngraph::Node>>& nodes, NodeMap& node_map)
{
    // for each node in topological order
    auto sorted_nodes = topological_sort(nodes);
    for (auto node : sorted_nodes)
    {
        if (!node_map.exists(node))
        {
            // get (already) cloned arguments and clone the node
            NodeVector cloned_args;
            for (auto arg : node->get_input_ops())
            {
                cloned_args.push_back(node_map.get(arg));
            }
            node_map.add(node, node->copy_with_new_args(cloned_args));
        }
    }

    // create and return list of cloned nodes
    // order matches input list (not necessarily topological)
    std::list<std::shared_ptr<ngraph::Node>> cloned_nodes;
    for (auto node : nodes)
    {
        cloned_nodes.push_back(node_map.get(node));
    }
    return cloned_nodes;
}

std::shared_ptr<ngraph::Function> ngraph::clone_function(std::shared_ptr<ngraph::Function> func,
                                                         NodeMap& node_map)
{
    // clone function operations
    clone_nodes(func->get_ops(), node_map);

    // get cloned function results and parameters
    ResultVector cloned_results;
    for (shared_ptr<Node> node : func->get_results())
    {
        auto result = std::dynamic_pointer_cast<op::Result>(node_map.get(node));
        if (!result)
        {
            throw ngraph_error("Results should be of type op::Result");
        }
        cloned_results.push_back(result);
    }
    std::vector<std::shared_ptr<op::Parameter>> cloned_params;
    for (auto param : func->get_parameters())
    {
        cloned_params.push_back(std::dynamic_pointer_cast<op::Parameter>(node_map.get(param)));
    }

    // create and return cloned function
    return std::make_shared<ngraph::Function>(cloned_results, cloned_params);
}

bool ngraph::is_equal_to_const_value(std::string const_value, std::shared_ptr<Node> reduce_constant)
{
    if (auto rc = dynamic_pointer_cast<ngraph::op::Constant>(reduce_constant))
    {
        auto cshape = rc->get_shape();
        size_t n = shape_size(cshape);
        // way to construct a constant of a given type, shape, value
        std::vector<std::string> vector_zero{n, const_value};
        auto constant_val_op =
            std::make_shared<ngraph::op::Constant>(rc->get_element_type(), cshape, vector_zero);

        // way to compare elements to const_value
        size_t n_bytes = n * rc->get_element_type().size();
        NGRAPH_DEBUG << "Comparing " << n_bytes << " bytes";
        return !memcmp(constant_val_op->get_data_ptr(), rc->get_data_ptr(), n_bytes);
    }
    else
    {
        return false;
    }
}

// Insert parameter node between src_node and dst_node by splitting the graph
// This function is not meant to be used by backends directly. Typically, a hybrid backend should
// only need to call ngraph::split_function_by_placement.
void ngraph::insert_parameter_split_between(shared_ptr<Node> src_node,
                                            shared_ptr<Node> dst_node,
                                            shared_ptr<op::Parameter> p_node)
{
    if (src_node->get_output_size() > 1)
    {
        throw ngraph_error("Node with more than one output not tested");
    }

    // Fix I/O. src_node and dst_node are connected via their input and output ports. We'll need
    // to cut those connections and connect p_node with dst_node with I/O ports.
    // Remove src_node's output to dst_node
    descriptor::Input* input = dst_node->get_input_from(src_node);
    descriptor::Output* output = src_node->get_output_to(dst_node);
    output->remove_input(input);
    // Change the corresponding dst_node's input's output to p_node's output
    input->replace_output(p_node, 0);

    // Fix users/arguments. src_node and dst_node are also connected by `users` and `args`. We'll
    // need to cut those connections and connect p_node with dst_node.
    // Remove dst_node from src_node's users
    const_cast<multiset<Node*>&>(src_node->users()).erase(dst_node.get());
    // Add dst_node to p_node's users
    const_cast<multiset<Node*>&>(p_node->users()).insert(dst_node.get());
    // Change dst_node's argument from src_node to p_node
    auto& args = const_cast<NodeVector&>(dst_node->get_arguments_FOR_GRAPH_REWRITE_ONLY());
    auto it = find(begin(args), end(args), src_node);
    if (it == end(args))
    {
        throw ngraph_error("src_node is not an input to dst_node");
    }
    it = args.erase(it);
    args.insert(it, p_node);
}

// Assert that nodes in the function is colocated and return that placement
Placement ngraph::get_colocated_function_placement(shared_ptr<Function> func)
{
    Placement function_placement = Placement::DEFAULT;
    traverse_nodes(func, [&](shared_ptr<Node> node) {
        Placement node_placement = node->get_placement();
        if (node_placement == Placement::DEFAULT)
        {
            throw ngraph_error("Node should have a device placement, not Placement::DEFAULT");
        }
        if (function_placement == Placement::DEFAULT)
        {
            // First time seeing a node
            function_placement = node->get_placement();
        }
        else if (function_placement != node_placement)
        {
            throw ngraph_error("Function contains nodes of two different placements");
        }
    });
    return function_placement;
}

// Helper function used by ngraph::split_function_by_placement to find the largest subgraph that
// can be placed on the same device starting from outputs and traversing towards inputs.
// Returns a function over that subgraph.
static shared_ptr<Function> build_largest_colocated_function(
    vector<shared_ptr<Node>> outputs,
    unordered_map<shared_ptr<op::Parameter>, shared_ptr<Node>>& map_parameter_to_source_node)
{
    // The outputs have the same placement, guaranteed by get_colocated_outputs_with_highest_orders
    if (outputs.size() == 0)
    {
        throw ngraph_error("Function has no output");
    }
    Placement function_placement = outputs[0]->get_placement();

    // Traverse to all reachable nodes from outputs
    deque<shared_ptr<Node>> stack;
    vector<shared_ptr<op::Parameter>> collected_parameters;
    unordered_set<shared_ptr<Node>> instances_seen;

    // A source_node S could be used by a downstream parameter P that is used by multiple downstream
    // ops. This avoids creating duplicated parameter nodes.
    unordered_map<shared_ptr<Node>, shared_ptr<op::Parameter>> map_source_node_to_parameter;

    for (auto output : outputs)
    {
        stack.push_front(output);
    }

    while (!stack.empty())
    {
        shared_ptr<Node> n = stack.front();
        stack.pop_front();

        if (instances_seen.count(n) == 0)
        {
            instances_seen.insert(n);

            if (n->description() == "Parameter")
            {
                auto n_prameter = static_pointer_cast<op::Parameter>(n);
                collected_parameters.push_back(n_prameter);
                if (map_parameter_to_source_node.find(n_prameter) ==
                    map_parameter_to_source_node.end())
                {
                    throw ngraph_error(
                        "Node " + n->get_name() +
                        " does not exist in map_parameter_to_source_node, while it must be a global"
                        "parameter and its own source node");
                }
            }

            for (auto input_op : n->get_input_ops())
            {
                if (input_op->get_placement() == function_placement)
                {
                    // If same placement the same, only add to stack if unseen
                    if (instances_seen.count(input_op) == 0)
                    {
                        stack.push_front(input_op);
                    }
                }
                else
                {
                    // Different placement. We don't put input_op on stack or instances_seen
                    if (map_source_node_to_parameter.find(input_op) ==
                        map_source_node_to_parameter.end())
                    {
                        shared_ptr<op::Parameter> p = make_shared<op::Parameter>(
                            input_op->get_output_element_type(0), input_op->get_output_shape(0));
                        p->set_placement(function_placement);
                        insert_parameter_split_between(input_op, n, p);
                        map_source_node_to_parameter[input_op] = p;
                        map_parameter_to_source_node[p] = input_op;
                        collected_parameters.push_back(p);
                    }
                    else
                    {
                        // Wire up p to n
                        shared_ptr<op::Parameter> p = map_source_node_to_parameter.at(input_op);
                        insert_parameter_split_between(input_op, n, p);
                    }
                }
            }
        }
    }
    auto func = make_shared<Function>(outputs, collected_parameters);

    /*
    for (size_t i = 0; i < outputs.size(); i++)
    {
        auto result = func->get_results().at(i);
        auto src_node = outputs.at(i);
        if (result->get_input_op(0) != src_node)
        {
            throw ngraph_error("new_key's input should be equal to output");
        }
        
        auto parm = map_source_node_to_parameter[src_node];
        map_parameter_to_source_node[parm] = result;
    }
    */
    return func;
}

// The returned nodes contains the node N with highest order. If N is placed at P, the returned
// nodes also include all other nodes placed at P while having orders larger than any non-P nodes.
static unordered_set<shared_ptr<Node>> get_colocated_outputs_with_highest_orders(
    const unordered_set<shared_ptr<Node>>& outputs,
    const unordered_map<shared_ptr<Node>, size_t>& map_node_to_order)
{
    shared_ptr<Node> highest_order_node;
    size_t highest_order = 0;
    for (auto node : outputs)
    {
        if (map_node_to_order.at(node) > highest_order)
        {
            highest_order = map_node_to_order.at(node);
            highest_order_node = node;
        }
    }

    Placement highest_order_node_placement = highest_order_node->get_placement();
    size_t highest_order_with_different_placement = 0;
    for (auto node : outputs)
    {
        if (node->get_placement() != highest_order_node_placement &&
            map_node_to_order.at(node) > highest_order_with_different_placement)
        {
            highest_order_with_different_placement = map_node_to_order.at(node);
        }
    }

    unordered_set<shared_ptr<Node>> colocated_outputs_with_highest_orders;
    for (auto node : outputs)
    {
        if (node->get_placement() == highest_order_node_placement &&
            map_node_to_order.at(node) > highest_order_with_different_placement)
        {
            colocated_outputs_with_highest_orders.insert(node);
        }
    }

    return colocated_outputs_with_highest_orders;
}

// Split function by placement, maximizing the span each subgraph. Each subgraph will be placed in
// a single device.
//
// - For nested functions, if one of `func` in `funcs` have a nested function `func_sub`, then
//     - `func_sub` was already guaranteed to have the same placement as `func`.
//     - `func_sub` may be shared by one or more `func`s in `funcs`, all of these `func`s is
//       guaranteed to use same placement
//     - `func_sub` will not show up the top list of `funcs`
// - The returned colocated functions will not have any kind of aliasing.
//     - (Input-output aliasing) If an input node is also a output node, this output node will not
//       be visited.
//     - (Constant-output aliasing) If an output node is a constant node, this output node will not
//       be visited as well.
//     - (Output aliasing) If a node appear multiple times in the outputs, it will only be visited
//       once.
//     - The user of split_function_by_placement must be aware of this property and insert copies
//       when applicable.
//
// TODO: split_function_by_placement may produce subgraphs with duplicated nodes. Refactor this to
//       edge-contraction + cycle detection algorithm to avoid this issue.
vector<shared_ptr<Function>> ngraph::split_function_by_placement(
    shared_ptr<Function> f,
    unordered_map<shared_ptr<op::Parameter>, shared_ptr<Node>>& map_parameter_to_source_node)
{
    // Store topological sorted orders for selecting output groups. If a node is used multiple
    // times, any order of the node will be valid since f is a DAG.
    unordered_map<shared_ptr<Node>, size_t> map_node_to_order;
    size_t node_idx = 1; // Starting from 1 s.t. the 0th index is smaller than any one of them
    for (auto node : f->get_ordered_ops())
    {
        map_node_to_order[node] = node_idx++;
    }

    // Initialize map_parameter_to_source_node map
    unordered_set<shared_ptr<Node>> f_parameters;
    for (auto parameter : f->get_parameters())
    {
        map_parameter_to_source_node[parameter] = parameter;
        f_parameters.insert(parameter);
    }

    // Using set to remove output aliasing
    unordered_set<shared_ptr<Node>> unvisited_outputs;
    for (auto node : f->get_results())
    {
        // Remove input-output and constant-output aliasing
        if (f_parameters.count(node) == 0 && node->description() != "Constant")
        {
            unvisited_outputs.insert(node->get_input_op(0));
        }
    }

    // Split outputs to groups
    vector<shared_ptr<Function>> colocated_functions;
    while (!unvisited_outputs.empty())
    {
        unordered_set<shared_ptr<Node>> colocated_outputs =
            get_colocated_outputs_with_highest_orders(unvisited_outputs, map_node_to_order);

        vector<shared_ptr<Node>> colocated_outputs_vector(colocated_outputs.begin(),
                                                          colocated_outputs.end());
        shared_ptr<Function> colocated_function = build_largest_colocated_function(
            colocated_outputs_vector, map_parameter_to_source_node);
        colocated_functions.push_back(colocated_function);

        // Construct new outputs for the next function
        for (auto parameter : colocated_function->get_parameters())
        {
            // If `parameter` is not a top-level parameter, and the source of `parameter` is
            // not a top-level parameter, it then `source_node` is the output of a upstream function
            auto source_node = map_parameter_to_source_node.at(parameter);
            if (f_parameters.find(parameter) == f_parameters.end() &&
                f_parameters.find(source_node) == f_parameters.end() &&
                source_node->description() != "Constant")
            {
                unvisited_outputs.insert(source_node);
            }
        }

        // Can not use std::set_difference since unordered
        unordered_set<shared_ptr<Node>> updated_unvisited_outputs;
        for (auto node : unvisited_outputs)
        {
            if (colocated_outputs.count(node) == 0)
            {
                updated_unvisited_outputs.insert(node);
            }
        }
        unvisited_outputs = updated_unvisited_outputs;
    }

    unordered_map<shared_ptr<Node>, shared_ptr<Node>> map_source_node_to_result;
    for (auto cf : colocated_functions)
    {
        for (auto r : cf->get_results())
        {
            map_source_node_to_result[r->get_input_op(0)] = r;
        }
    }

    for (auto it = map_parameter_to_source_node.begin(); it != map_parameter_to_source_node.end();
         ++it)
    {
        if (map_source_node_to_result.count(it->second) != 0)
        {
            it->second = map_source_node_to_result[it->second];
        }
    }

    // The colocated_functions should be called in reversed order
    reverse(colocated_functions.begin(), colocated_functions.end());
    return colocated_functions;
}
