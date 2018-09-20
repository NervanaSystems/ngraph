//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/result_vector.hpp"
#include "ngraph/placement.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

void ngraph::traverse_nodes(const std::shared_ptr<const Function> p,
                            std::function<void(std::shared_ptr<Node>)> f,
                            bool include_control_deps)
{
    traverse_nodes(p.get(), f, include_control_deps);
}

void ngraph::traverse_nodes(const Function* p,
                            std::function<void(std::shared_ptr<Node>)> f,
                            bool include_control_deps)
{
    NodeVector nodes;

    for (auto r : p->get_results())
    {
        nodes.push_back(r);
    }

    for (auto param : p->get_parameters())
    {
        nodes.push_back(param);
    }

    traverse_nodes(nodes, f, include_control_deps);
}

// This version of traverses directly from input/output nodes to perform functions on
// graphs that are not wrapped by functions. Most useful for finding parameters of a graph
// directly from the result nodes, not from function parameters.
void ngraph::traverse_nodes(const NodeVector& io_nodes,
                            std::function<void(std::shared_ptr<Node>)> f,
                            bool include_control_deps)
{
    std::unordered_set<std::shared_ptr<Node>> instances_seen;
    std::deque<std::shared_ptr<Node>> stack;

    for (auto r : io_nodes)
    {
        stack.push_front(r);
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
        for (auto arg : n->get_arguments())
        {
            if (instances_seen.count(arg) == 0)
            {
                stack.push_front(arg);
            }
        }

        if (include_control_deps)
        {
            for (auto cdep : n->get_control_dependencies())
            {
                if (instances_seen.count(cdep) == 0)
                {
                    stack.push_front(cdep);
                }
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

void ngraph::replace_node(std::shared_ptr<Node> target, std::shared_ptr<Node> replacement)
{
    if (target->is_output())
    {
        throw ngraph_error("Result nodes cannot be replaced.");
    }

    if (target->get_users().empty())
    {
        throw ngraph_error("replacing an unreachable node");
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
}

// Check if all paths from X to a result go through Y
bool ngraph::is_post_dominated(Node* X, Node* Y)
{
    std::unordered_set<Node*> visited;
    std::deque<Node*> stack;
    stack.push_front(X);

    while (stack.size() > 0)
    {
        ngraph::Node* curr = stack.front();
        visited.insert(curr);
        if (curr->is_output())
        {
            return false;
        }
        stack.pop_front();
        if (curr != Y)
        {
            for (auto next : curr->get_users())
            {
                if (visited.count(next.get()) == 0)
                {
                    stack.push_front(next.get());
                }
            }
        }
    }
    return true;
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
    auto sorted_nodes = topological_sort(nodes, true);
    for (auto node : sorted_nodes)
    {
        if (!node_map.exists(node))
        {
            // get (already) cloned arguments and clone the node
            NodeVector cloned_args;
            for (auto arg : node->get_arguments())
            {
                cloned_args.push_back(node_map.get(arg));
            }
            auto cloned_node = node->copy_with_new_args(cloned_args);

            //copy control dependencies
            for (auto cdep : node->get_control_dependencies())
            {
                cloned_node->add_control_dependency(node_map.get(cdep));
            }
            node_map.add(node, cloned_node);
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

std::shared_ptr<ngraph::Function> ngraph::clone_function(const ngraph::Function& func)
{
    NodeMap nm;
    return clone_function(func, nm);
}

std::shared_ptr<ngraph::Function> ngraph::clone_function(const ngraph::Function& func,
                                                         NodeMap& node_map)
{
    // clone function operations
    clone_nodes(func.get_ops(true), node_map);

    // get cloned function results and parameters
    ResultVector cloned_results;
    for (shared_ptr<Node> node : func.get_results())
    {
        auto result = std::dynamic_pointer_cast<op::Result>(node_map.get(node));
        if (!result)
        {
            throw ngraph_error("Results should be of type op::Result");
        }
        cloned_results.push_back(result);
    }
    std::vector<std::shared_ptr<op::Parameter>> cloned_params;
    for (auto param : func.get_parameters())
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

// Insert result and parameter node between src_node and dst_node by splitting the graph
//
// Before:                        |  After:
// (Device:0)         (Device:1)  |  (Device:0)         (Device:0)  (Device:1)         (Device:1)
// +-----+---+       +---+-----+  |  +-----+---+       +---+-----+  +-----+---+       +---+-----+
// |     |   |       |   |     |  |  |     |   |       |   |     |  |     |   |       |   |     |
// |     | o +--[0]--> i |     |  |  |     | o +--[4]--> i |     |  |     | o +--[8]--> i |     |
// |     |   <--[1]--+   |     |  |  |     |   <--[5]--+   |     |  |     |   <--[9]--+   |     |
// | src +---+       +---+ dst |  |  | src +---+       +---+ res |  | par +---+       +---+ dst |
// |     |               |     |  |  |     |               |     |  |     |               |     |
// |     +------[2]------>     |  |  |     +------[6]------>     |  |     +------[10]----->     |
// |     <------[3]------+     |  |  |     <------[7]------+     |  |     <------[11]-----+     |
// +-----+               +-----+  |  +-----+               +-----+  +-----+               +-----+
pair<shared_ptr<op::Result>, shared_ptr<op::Parameter>>
    ngraph::insert_result_parameter_split(const shared_ptr<Node>& src_node,
                                          const shared_ptr<Node>& dst_node)
{
    if (src_node->get_output_size() != 1)
    {
        throw ngraph_error("Multiple output per op not supported in graph partition yet.");
    }

    // Make parameter node
    shared_ptr<op::Parameter> par_node = make_shared<op::Parameter>(
        src_node->get_output_element_type(0), src_node->get_output_shape(0));
    par_node->set_placement(dst_node->get_placement());

    // Fix input / output among src, dst and par
    descriptor::Input* dst_input = dst_node->get_input_from(src_node);
    descriptor::Output* src_output = src_node->get_output_to(dst_node);
    src_output->remove_input(dst_input);    // Remove [0]
    dst_input->replace_output(par_node, 0); // Remove [0] (again), add [8], remove [1], add [9]

    // Add res node
    shared_ptr<op::Result> res_node = make_shared<op::Result>(src_node); // Add [4], [5], [6], [7]
    res_node->set_placement(src_node->get_placement());

    return make_pair(res_node, par_node);
}

// Insert unary node between two nodes like S->D => S->N->D
// Before:                        |  After:
// +-----+---+       +---+-----+  |  +-----+---+       +---+-----+---+       +---+-----+
// |     |   |       |   |     |  |  |     |   |       |   |     |   |       |   |     |
// |     | o +--[0]--> i |     |  |  |     | o +--[4]--> i |     | o +--[8]--> i |     |
// |     |   <--[1]--+   |     |  |  |     |   <--[5]--+   |     |   <--[9]--+   |     |
// | src +---+       +---+ dst |  |  | src +---+       +---+ new +---+       +---+ dst |
// |     |               |     |  |  |     |               |     |               |     |
// |     +------[2]------>     |  |  |     +------[6]------>     +------[10]----->     |
// |     <------[3]------+     |  |  |     <------[7]------+     <------[11]-----+     |
// +-----+               +-----+  |  +-----+               +-----+               +-----+
//                                |
// +-----+---+       +---+-----+  |
// |     |   |       |   |     |  |
// |     | o +--[4]--> i |     |  |
// |     |   <--[5]--+   |     |  |
// | src +---+       +---+ new |  |
// |     |               |     |  |
// |     +------[6]------>     |  |
// |     <------[7]------+     |  |
// +-----+               +-----+  |
//
// This cannot be achieved by ngraph::replace_node().
// With replace_node(), we could do:
// [     S           S      ]
// [    / \          |      ]
// [   /   \   =>    N      ]
// [  /     \       / \     ]
// [ D0     D1    D0   D1   ]
//
// But we want:
// [     S            S     ]
// [    / \          / \    ]
// [   /   \   =>   N0  N1  ]
// [  /     \      /     \  ]
// [ D0     D1    D0     D1 ]
//
// Typically new_node is connected to src_node already. The reason we don't create `new_node`
// inside the function and return it (similar to ngraph::insert_result_parameter_split) is that
// we'll have to templatize its function to call new_node's constructor.
void ngraph::insert_new_node_between(const shared_ptr<Node>& src_node,
                                     const shared_ptr<Node>& dst_node,
                                     const shared_ptr<Node>& new_node)
{
    // Fix input / output
    descriptor::Input* dst_input = dst_node->get_input_from(src_node);
    descriptor::Output* src_output = src_node->get_output_to(dst_node);
    src_output->remove_input(dst_input);    // Remove [0]
    dst_input->replace_output(new_node, 0); // Remove [0] (again), add [8], remove [1], add [9]
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

std::shared_ptr<Node> ngraph::make_zero(const element::Type& element_type, const Shape& shape)
{
    std::shared_ptr<Node> zero = op::Constant::create(element_type, Shape{}, {0.0});
    if (shape.size() > 0)
    {
        AxisSet axes;
        for (size_t i = 0; i < shape.size(); i++)
        {
            axes.insert(i);
        }
        zero = std::make_shared<op::Broadcast>(zero, shape, axes);
    }
    return zero;
}

std::shared_ptr<Node> ngraph::make_constant_from_string(std::string val,
                                                        const element::Type& element_type,
                                                        const Shape& shape)
{
    auto cvals = std::vector<std::string>(shape_size(shape), val);
    return std::make_shared<op::Constant>(element_type, shape, cvals);
}

bool ngraph::is_zero(std::shared_ptr<Node> reduce_constant)
{
    auto result_bool = is_equal_to_const_value("0", reduce_constant);
    return result_bool;
}

bool ngraph::is_one(std::shared_ptr<Node> reduce_constant)
{
    auto result_bool = is_equal_to_const_value("1", reduce_constant);
    return result_bool;
}

NodeVector ngraph::get_subgraph_outputs(const NodeVector& nodes,
                                        const NodeVector& exclusions,
                                        bool ignore_unused)
{
    std::set<shared_ptr<Node>> exclusions_set(exclusions.begin(), exclusions.end());
    std::set<shared_ptr<Node>> nodes_set(nodes.begin(), nodes.end());

    NodeVector outputs;

    for (auto n : nodes)
    {
        if (exclusions_set.count(n) != 0)
        {
            continue;
        }

        for (auto u : n->get_users())
        {
            if (nodes_set.count(u) == 0 && (!ignore_unused || is_used(u.get())))
            {
                outputs.push_back(n);
            }
        }
    }
    return outputs;
}

bool ngraph::is_used(Node* node)
{
    std::unordered_set<Node*> instances_seen;
    std::deque<Node*> stack;
    stack.push_front(node);

    while (stack.size() > 0)
    {
        ngraph::Node* n = stack.front();
        if (instances_seen.count(n) == 0)
        {
            if (n->is_output())
            {
                return true;
            }
            instances_seen.insert(n);
        }
        stack.pop_front();
        for (auto arg : n->get_users())
        {
            if (instances_seen.count(arg.get()) == 0)
            {
                stack.push_front(arg.get());
            }
        }
    }
    return false;
}

size_t ngraph::get_user_count(Node* node)
{
    size_t count = 0;
    for (auto node_user : node->get_users())
    {
        count += is_used(node_user.get());
    }
    return count;
}

bool ngraph::possibly_overwritten(Node* node)
{
    for (const descriptor::Output& output : node->get_outputs())
    {
        for (const descriptor::Input* input : output.get_inputs())
        {
            if (auto op = std::dynamic_pointer_cast<ngraph::op::Op>(input->get_node()))
            {
                if (auto op_annotations = op->get_op_annotations())
                {
                    for (auto oi_pair : op_annotations->get_in_place_oi_pairs())
                    {
                        if (input->get_index() == oi_pair.input && oi_pair.destructive)
                        {
                            return true;
                        }
                    }
                }
            }
        }
    }
    return false;
}

bool ngraph::is_strided(const Strides& strides)
{
    return std::any_of(strides.begin(), strides.end(), [](size_t stride) { return stride != 1; });
}
