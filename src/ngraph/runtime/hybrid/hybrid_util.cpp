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

#include "ngraph/runtime/hybrid/hybrid_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"

using namespace ngraph;
using namespace std;

static Node* take_independent_node_with_placement_priority(
    map<size_t, deque<Node*>>& independent_nodes_by_placement, size_t placement)
{
    Node* selected_node = nullptr;
    auto it = independent_nodes_by_placement.find(placement);
    if (it != independent_nodes_by_placement.end() && it->second.size() != 0)
    {
        selected_node = it->second.front();
        it->second.pop_front();
    }
    else
    {
        for (auto& p : independent_nodes_by_placement)
        {
            if (p.second.size() > 0)
            {
                selected_node = p.second.front();
                p.second.pop_front();
                break;
            }
        }
    }
    return selected_node;
}

static vector<unordered_set<shared_ptr<Node>>>
    group_function_nodes_to_clusters(const shared_ptr<Function>& f)
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
    while (Node* independent_node = ::take_independent_node_with_placement_priority(
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

static map<shared_ptr<op::Result>, shared_ptr<op::Parameter>>
    insert_result_parameter_split(const shared_ptr<Node>& src_node,
                                  const shared_ptr<Node>& dst_node)
{
    map<shared_ptr<op::Result>, shared_ptr<op::Parameter>> result_map;

    for (descriptor::Input& input : dst_node->get_inputs())
    {
        if (input.get_output().get_node() == src_node)
        {
            descriptor::Input* dst_input = &input;
            descriptor::Output* src_output = &input.get_output();

            // Make parameter node
            shared_ptr<op::Parameter> par_node =
                make_shared<op::Parameter>(src_output->get_element_type(), src_output->get_shape());
            par_node->set_placement_index(dst_node->get_placement_index());

            // Fix input / output among src, dst and par
            // Remove [0]
            src_output->remove_input(dst_input);

            // Remove [0] (again), add [8], remove [1], add [9]
            dst_input->replace_output(par_node, 0);

            // Add res node
            shared_ptr<op::Result> res_node =
                make_shared<op::Result>(src_node); // Add [4], [5], [6], [7]
            res_node->set_placement_index(src_node->get_placement_index());

            result_map.insert({res_node, par_node});
        }
    }
    return result_map;
}

//  will be removed when the backends move to the latest Hybrid backend
pair<vector<shared_ptr<Function>>, unordered_map<shared_ptr<op::Parameter>, shared_ptr<op::Result>>>
    runtime::hybrid::split_function_by_placement(const shared_ptr<Function>& f)
{
    // Split functions to clusters of nodes that can be computed together
    vector<unordered_set<shared_ptr<Node>>> clusters = ::group_function_nodes_to_clusters(f);

    // Map from (intermediate) parameter to result node, for guiding data copy among devices
    unordered_map<shared_ptr<op::Parameter>, shared_ptr<op::Result>> map_parameter_to_result;

    // for (auto x : f->get_parameters())
    // {
    //     NGRAPH_INFO << x->get_name();
    // }
    // cluster zero ends up with all of the Parameters, even if they are not needed in cluster
    // zero. Remove all Parameters from cluster zero.
    unordered_set<shared_ptr<Node>> new_cluster;
    for (auto node : clusters[0])
    {
        if (node->description() != "Parameter")
        {
            new_cluster.insert(node);
        }
    }
    clusters[0].swap(new_cluster);
    for (auto node : clusters[0])
    {
        NGRAPH_INFO << node->get_name();
    }

    // Split neighboring nodes if they belong to different clusters
    // TODO: optimization to group multiple result node from the same source,
    //       and to group the parameter node in the same cluster with the same result node source
    unordered_map<shared_ptr<Node>, unordered_set<shared_ptr<Node>>*> map_node_to_cluster;
    for (auto& cluster : clusters)
    {
        NGRAPH_INFO << "***************** new cluster";
        for (auto node : cluster)
        {
            NGRAPH_INFO << node->get_name();
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
                NGRAPH_INFO << "src=" << src_node->description()
                            << ", dst=" << dst_node->description();
                // Split src_node and dst_node
                map<shared_ptr<op::Result>, shared_ptr<op::Parameter>> res_par_pair_map =
                    ::insert_result_parameter_split(src_node, dst_node);
                for (const auto& res_par_pair : res_par_pair_map)
                {
                    shared_ptr<op::Result> res_node = res_par_pair.first;
                    shared_ptr<op::Parameter> par_node = res_par_pair.second;
                    map_parameter_to_result[par_node] = res_node;

                    // Insert newly created nodes into clusters
                    src_cluster->insert(res_node);
                    dst_cluster->insert(par_node);
                }
            }
        }
    }

    // Create functions from clusters
    vector<shared_ptr<Function>> sub_functions;
    for (auto cluster : clusters)
    {
        ParameterVector par_vector;
        ResultVector res_vector;
        size_t placement = -1;
        for (auto node : cluster)
        {
            placement = node->get_placement_index();
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
        sub_function->set_placement(placement);
        sub_functions.push_back(sub_function);
#ifdef HYBRID_DEBUG
        ngraph::pass::Manager pass_manager;
        pass_manager.register_pass<ngraph::pass::VisualizeTree>("subgraph_" + to_string(index++) +
                                                                ".png");
        pass_manager.run_passes(sub_function);
#endif
    }

    return make_pair(sub_functions, map_parameter_to_result);
}
