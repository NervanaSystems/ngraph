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

#include "ngraph/runtime/hybrid/hybrid_util.hpp"

using namespace ngraph;
using namespace std;

static Node* take_independent_node_with_placement_priority_size(
    map<size_t, deque<Node*>>& independent_nodes_by_placement, size_t placement)
{
    Node* selected_node = nullptr;
    if (independent_nodes_by_placement.find(placement) != independent_nodes_by_placement.end() &&
        independent_nodes_by_placement.at(placement).size() != 0)
    {
        selected_node = independent_nodes_by_placement.at(placement).front();
        independent_nodes_by_placement.at(placement).pop_front();
    }
    else
    {
        for (auto& it : independent_nodes_by_placement)
        {
            if (it.second.size() > 0)
            {
                selected_node = it.second.front();
                it.second.pop_front();
                break;
            }
        }
    }
    return selected_node;
}

static vector<unordered_set<shared_ptr<Node>>>
    group_function_nodes_to_clusters_size(const shared_ptr<Function>& f)
{
    // Topologically sort nodes by picking independent node with the same placement as the
    // previously picked node greedily
    map<size_t, deque<Node*>> independent_nodes_by_placement;
    unordered_map<Node*, size_t> node_dependency_count;
    unordered_map<ngraph::Node*, shared_ptr<ngraph::Node>> node_map;

    for (shared_ptr<Node> node : f->get_ops())
    {
        size_t dependency_count = node->get_arguments().size();
        node_map[node.get()] = node;
        node_dependency_count[node.get()] = dependency_count;
        if (dependency_count == 0)
        {
            independent_nodes_by_placement[node->get_placement_index()].push_back(node.get());
        }
    }

    list<shared_ptr<Node>> sorted_nodes;
    size_t previous_placement = 0;
    while (Node* independent_node = take_independent_node_with_placement_priority_size(
               independent_nodes_by_placement, previous_placement))
    {
        previous_placement = independent_node->get_placement_index();
        sorted_nodes.push_back(node_map.at(independent_node));

        for (auto user : independent_node->get_users())
        {
            Node* user_node = user.get();
            node_dependency_count.at(user_node) -= 1;
            if (node_dependency_count.at(user_node) == 0)
            {
                independent_nodes_by_placement[user_node->get_placement_index()].push_back(
                    user_node);
            }
        }
    }

    if (sorted_nodes.size() != f->get_ops().size())
    {
        throw ngraph_error("sorted_nodes.size()== " + to_string(sorted_nodes.size()) +
                           " != f->get_ops().size()== " + to_string(f->get_ops().size()) +
                           ". Internal error with topological sort.");
    }

    // Build clusters from the sorted_nodes
    previous_placement = Node::placement_invalid;
    vector<unordered_set<shared_ptr<Node>>> clusters;
    for (shared_ptr<Node> node : sorted_nodes)
    {
        size_t node_placement = node->get_placement_index();
        if (node_placement != previous_placement)
        {
            unordered_set<shared_ptr<Node>> new_cluster;
            clusters.push_back(new_cluster);
        }
        clusters.back().insert(node);
        previous_placement = node_placement;
    }

    // Sanity check for node duplication and full node coverage
    unordered_set<shared_ptr<Node>> cluster_nodes;
    for (auto cluster : clusters)
    {
        for (auto node : cluster)
        {
            if (cluster_nodes.find(node) != cluster_nodes.end())
            {
                throw ngraph_error("Node " + node->get_name() + " is duplicated in clusters");
            }
            cluster_nodes.insert(node);
        }
    }
    unordered_set<shared_ptr<Node>> f_nodes;
    for (auto node : f->get_ordered_ops())
    {
        f_nodes.insert(node);
    }
    if (cluster_nodes != f_nodes)
    {
        throw ngraph_error(
            "Cluster's nodes are not the same as function's nodes. cluster_nodes.size()=" +
            to_string(cluster_nodes.size()) + ", f_nodes.size()=" + to_string(f_nodes.size()));
    }

    return clusters;
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

// Suffix *_size  as a part of function name is temporary, this suffix
//  will be removed when the backends move to the latest Hybrid backend
pair<shared_ptr<op::Result>, shared_ptr<op::Parameter>>
    insert_result_parameter_split_size(const shared_ptr<Node>& src_node,
                                       const shared_ptr<Node>& dst_node)
{
    if (src_node->get_output_size() != 1)
    {
        throw ngraph_error("Multiple output per op not supported in graph partition yet.");
    }

    // Make parameter node
    shared_ptr<op::Parameter> par_node = make_shared<op::Parameter>(
        src_node->get_output_element_type(0), src_node->get_output_shape(0));
    par_node->set_placement_index(dst_node->get_placement_index());

    // Fix input / output among src, dst and par
    descriptor::Input* dst_input = dst_node->get_input_from(src_node);
    descriptor::Output* src_output = src_node->get_output_to(dst_node);
    src_output->remove_input(dst_input);    // Remove [0]
    dst_input->replace_output(par_node, 0); // Remove [0] (again), add [8], remove [1], add [9]

    // Add res node
    shared_ptr<op::Result> res_node = make_shared<op::Result>(src_node); // Add [4], [5], [6], [7]
    res_node->set_placement_index(src_node->get_placement_index());

    return make_pair(res_node, par_node);
}

// Suffix *_size  as a part of function name is temporary, this suffix
//  will be removed when the backends move to the latest Hybrid backend
pair<vector<shared_ptr<Function>>, unordered_map<shared_ptr<op::Parameter>, shared_ptr<op::Result>>>
    runtime::hybrid::split_function_by_placement_size(const shared_ptr<Function>& f)
{
    // Split functions to clusters of nodes that can be computed together
    vector<unordered_set<shared_ptr<Node>>> clusters = group_function_nodes_to_clusters_size(f);

    // Map from (intermediate) parameter to result node, for guiding data copy among devices
    unordered_map<shared_ptr<op::Parameter>, shared_ptr<op::Result>> map_parameter_to_result;

    // Split neighboring nodes if they belong to different clusters
    // TODO: optimization to group multiple result node from the same source,
    //       and to group the parameter node in the same cluster with the same result node source
    unordered_map<shared_ptr<Node>, unordered_set<shared_ptr<Node>>*> map_node_to_cluster;
    for (auto& cluster : clusters)
    {
        for (auto node : cluster)
        {
            map_node_to_cluster[node] = &cluster;
        }
    }
    for (auto dst_node : f->get_ordered_ops())
    {
        for (auto src_node : dst_node->get_arguments())
        {
            auto src_cluster = map_node_to_cluster.at(src_node);
            auto dst_cluster = map_node_to_cluster.at(dst_node);
            if (src_cluster != dst_cluster)
            {
                // Split src_node and dst_node
                pair<shared_ptr<op::Result>, shared_ptr<op::Parameter>> res_par_pair =
                    insert_result_parameter_split_size(src_node, dst_node);
                shared_ptr<op::Result> res_node = res_par_pair.first;
                shared_ptr<op::Parameter> par_node = res_par_pair.second;
                map_parameter_to_result[par_node] = res_node;

                // Insert newly created nodes into clusters
                src_cluster->insert(res_node);
                dst_cluster->insert(par_node);
            }
        }
    }

    // Create functions from clusters
    vector<shared_ptr<Function>> sub_functions;
    for (auto cluster : clusters)
    {
        ParameterVector par_vector;
        ResultVector res_vector;
        for (auto node : cluster)
        {
            if (auto res_node = dynamic_pointer_cast<op::Result>(node))
            {
                res_vector.push_back(res_node);
            }
            else if (auto par_node = dynamic_pointer_cast<op::Parameter>(node))
            {
                par_vector.push_back(par_node);
            }
        }
        auto sub_function = make_shared<Function>(res_vector, par_vector);
        sub_functions.push_back(sub_function);
    }

    return make_pair(sub_functions, map_parameter_to_result);
}

// Suffix *_size  as a part of function name is temporary, this suffix
//  will be removed when the backends move to the latest Hybrid backend
// Assert that nodes in the function is colocated and return that placement
size_t runtime::hybrid::get_colocated_function_placement_size(shared_ptr<Function> func)
{
    auto ops = func->get_ops();

    //it's okay to not do Placement::DEFAULT check; the same node will be checked in the loop below
    size_t function_placement = ops.front()->get_placement_index();
    for (auto op : ops)
    {
        size_t node_placement = op->get_placement_index();
        if (node_placement == Node::placement_invalid)
        {
            throw ngraph_error("Node should have a device placement");
        }
        if (function_placement != node_placement)
        {
            throw ngraph_error("Function contains nodes of two different placements");
        }
    }

    return function_placement;
}
