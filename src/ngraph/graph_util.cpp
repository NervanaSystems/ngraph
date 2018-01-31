// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <algorithm>
#include <cassert>
#include <deque>
#include <forward_list>
#include <iomanip>
#include <map>
#include <unordered_set>
#include <vector>

#include "ngraph/common.hpp"
#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "ngraph/ops/parameter.hpp"

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

void ngraph::replace_node(std::shared_ptr<Node> target,
                          std::shared_ptr<Node> replacement,
                          bool replace_output)
{
    if (target->is_output() && !replace_output)
    {
        return;
    }

    //fix input/output descriptors
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
        auto& args = const_cast<ngraph::Nodes&>(user->get_arguments_FOR_GRAPH_REWRITE_ONLY());
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
            Nodes cloned_args;
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
    Nodes cloned_results;
    for (shared_ptr<Node> node : func->get_results())
    {
        cloned_results.push_back(node_map.get(node));
    }
    std::vector<std::shared_ptr<op::Parameter>> cloned_params;
    for (auto param : func->get_parameters())
    {
        cloned_params.push_back(std::dynamic_pointer_cast<op::Parameter>(node_map.get(param)));
    }

    // create and return cloned function
    return std::make_shared<ngraph::Function>(cloned_results, cloned_params);
}

// Insert parameter node between src_node and dst_node by splitting the graph
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
    auto& args = const_cast<Nodes&>(dst_node->get_arguments_FOR_GRAPH_REWRITE_ONLY());
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
    Placement placement = Placement::DEFAULT;
    traverse_nodes(func, [&](shared_ptr<Node> node) {
        if (node->description() == "Parameter")
        {
            if (node->get_placement() != Placement::DEFAULT)
            {
                throw ngraph_error("Parameter node must have DEFAULT placement");
            }
        }
        else
        {
            if (placement == Placement::DEFAULT)
            {
                placement = node->get_placement();
            }
            if (placement != node->get_placement())
            {
                throw ngraph_error("Function contains nodes of two different placements");
            }
        }
    });
    return placement;
}

// Helper function used by ngraph::split_function_by_placement to find the largest subgraph that
// can be placed on the same device starting from outputs and traversing towards inputs.
// Returns a function over that subgraph.
shared_ptr<Function> ngraph::build_largest_colocated_function(
    vector<shared_ptr<Node>> outputs,
    unordered_map<shared_ptr<Node>, shared_ptr<Node>>& map_node_to_source_node)
{
    if (outputs.size() == 0)
    {
        // TODO: handle zero-sized output when needed
        throw ngraph_error("Function has no output");
    }

    // Outputs shall all have same placement except parameter as output
    Placement function_placement = Placement::DEFAULT;
    for (auto output : outputs)
    {
        if (output->description() == "Parameter")
        {
            if (output->get_placement() != Placement::DEFAULT)
            {
                throw ngraph_error("Parameter must have DEFAULT placement");
            }
        }
        else
        {
            if (function_placement == Placement::DEFAULT)
            {
                function_placement = output->get_placement();
            }
            if (output->get_placement() != function_placement)
            {
                throw ngraph_error(
                    "Function placement invalid, all nodes in the same function should be place on "
                    "the same device.");
            }
        }
    }

    // If all nodes are parameters, just return that function
    if (function_placement == Placement::DEFAULT)
    {
        vector<shared_ptr<op::Parameter>> outputs_as_parameters;
        for (auto output : outputs)
        {
            map_node_to_source_node[output] = output;
            outputs_as_parameters.push_back(static_pointer_cast<op::Parameter>(output));
        }

        return make_shared<Function>(outputs, outputs_as_parameters);
    }

    // Traverse to all reachable nodes from outputs
    deque<shared_ptr<Node>> stack;
    vector<shared_ptr<op::Parameter>> collected_parameters;
    unordered_set<shared_ptr<Node>> instances_seen;
    unordered_map<shared_ptr<Node>, shared_ptr<op::Parameter>> map_source_node_to_parameter;

    for (auto output : outputs)
    {
        stack.push_front(output);
    }

    while (!stack.empty())
    {
        shared_ptr<Node> n = stack.front();

        if (instances_seen.count(n) == 0)
        {
            instances_seen.insert(n);
        }
        stack.pop_front();

        if (n->description() == "Parameter")
        {
            if (n->get_placement() != Placement::DEFAULT)
            {
                throw ngraph_error("Parameter must have DEFAULT placement");
            }
            collected_parameters.push_back(static_pointer_cast<op::Parameter>(n));
            map_node_to_source_node[n] = n; // Can add sanity check here
        }
        else
        {
            for (auto input_op : n->get_input_ops())
            {
                if (input_op->get_placement() == function_placement ||
                    input_op->get_placement() == Placement::DEFAULT)
                {
                    // If same placement the same, only add to stack if unseen, as we only need to
                    // visit all nodes but not all edges
                    if (instances_seen.count(input_op) == 0)
                    {
                        stack.push_front(input_op);
                    }
                }
                else
                {
                    // Different placement. We don't put them on stack or instances_seen.
                    if (map_source_node_to_parameter.find(input_op) ==
                        map_source_node_to_parameter.end())
                    {
                        shared_ptr<op::Parameter> p = make_shared<op::Parameter>(
                            input_op->get_output_element_type(0), input_op->get_output_shape(0));
                        insert_parameter_split_between(input_op, n, p);
                        map_source_node_to_parameter[input_op] = p;
                        map_node_to_source_node[p] = input_op;
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
    return func;
}

// Split function by placement, maximizing the span each subgraph. Each subgraph will be placed in
// a single device.
//
// - For nested functions, if one of `func` in `funcs` have a nested function `func_sub`, then
//     - `func_sub` was already guaranteed to have the same placement as `func`.
//     - `func_sub` may be shared by one or more `func`s in `funcs`, all of these `func`s is
//       guaranteed to use same placement
//     - `func_sub` will not show up the top list of `funcs`
// - Allowing 3 or more devices in the same graph can be reduced to supporting functions
//   with ouput op in 2 different devices, and vice versa. Now, we enforce that function's output
//   in the same device.
vector<shared_ptr<Function>> ngraph::split_function_by_placement(
    shared_ptr<Function> f,
    unordered_map<shared_ptr<Node>, shared_ptr<Node>>& map_node_to_source_node)
{
    // Initialize map_node_to_source_node map
    auto& f_parameters = f->get_parameters();
    for (size_t i = 0; i < f_parameters.size(); ++i)
    {
        map_node_to_source_node[f_parameters[i]] = f_parameters[i];
    }

    // Build
    vector<shared_ptr<Function>> funcs;
    vector<shared_ptr<Node>> outputs = f->get_results();

    while (true)
    {
        // Build current function from outputs
        shared_ptr<Function> func =
            build_largest_colocated_function(outputs, map_node_to_source_node);
        funcs.push_back(func);

        // Construct new outputs for the next function
        outputs.clear();
        for (auto param : func->get_parameters())
        {
            if (map_node_to_source_node.at(param) != param)
            {
                outputs.push_back(map_node_to_source_node.at(param));
            }
        }
        if (outputs.empty())
        {
            break;
        }
    }

    // Need to reverse
    reverse(funcs.begin(), funcs.end());
    return funcs;
}
