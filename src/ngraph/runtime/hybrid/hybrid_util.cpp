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
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/runtime/hybrid/op/function_call.hpp"

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

        for (auto user : independent_node->get_users(true))
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
    clusters.push_back(unordered_set<shared_ptr<Node>>());

    for (shared_ptr<Node> node : sorted_nodes)
    {
        size_t node_placement = node->get_placement_index();
        if (node_placement == 0)
        {
            clusters[0].insert(node);
        }
        else
        {
            if (node_placement != previous_placement)
            {
                unordered_set<shared_ptr<Node>> new_cluster;
                clusters.push_back(new_cluster);
            }
            clusters.back().insert(node);
        }
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

void runtime::hybrid::rewrite_function(const shared_ptr<Function>& f,
                                       const vector<shared_ptr<runtime::Backend>>& backend_list)
{
    // Split functions to clusters of nodes that can be computed together
    vector<unordered_set<shared_ptr<Node>>> clusters = ::group_function_nodes_to_clusters(f);

    // unordered_map<shared_ptr<op::Parameter>, shared_ptr<op::Result>> map_parameter_to_result;
    unordered_map<shared_ptr<Node>, unordered_set<shared_ptr<Node>>*> map_node_to_cluster;
    for (auto& cluster : clusters)
    {
        if (cluster.size() > 0)
        {
            shared_ptr<Node> tmp_node = *cluster.begin();
            if (tmp_node == nullptr)
            {
                throw runtime_error("cluster contains nullptr instead of nodes");
            }
            auto placement = tmp_node->get_placement_index();
            if (placement != 0)
            {
                // This is a non-native cluster so make it a FunctionCall
                vector<shared_ptr<Node>> function_call_inputs;
                vector<shared_ptr<Node>> function_call_outputs;
                ParameterVector cluster_inputs;
                NodeVector cluster_outputs;
                for (auto node : cluster)
                {
                    for (auto input : node->get_arguments())
                    {
                        if (input->get_placement_index() == 0 && !input->is_constant())
                        {
                            // Since this input is from outside the cluster we need to create
                            // a new Parameter node placed in the cluster instead of this external
                            // node. Constant nodes are ignored here since the values are available
                            // in the graph.
                            std::vector<Output<Node>> source_outputs =
                                get_outputs_to(*input, *node);
                            NGRAPH_CHECK(
                                source_outputs.size() == 1,
                                "rewrite_function encountered more than "
                                "one output between a cluster node and one of its arguments");
                            auto& source_output = source_outputs[0];

                            std::vector<Input<Node>> target_inputs = get_inputs_from(*input, *node);
                            NGRAPH_CHECK(
                                target_inputs.size() == 1,
                                "rewrite_function encountered more than "
                                "one input between a cluster node and one of its arguments");
                            auto& target_input = target_inputs[0];

                            auto new_parameter = make_shared<ngraph::op::Parameter>(
                                source_output.get_element_type(), source_output.get_shape());
                            new_parameter->set_placement_index(placement);
                            target_input.replace_source_output(new_parameter->output(0));
                            cluster_inputs.push_back(new_parameter);
                            function_call_inputs.push_back(input);
                        }
                    }
                    for (auto output : node->get_users(true))
                    {
                        if (output->get_placement_index() == 0)
                        {
                            // Since this output is to outside the cluster we need to create
                            // a new Result node placed in the cluster instead of this external
                            // node
                            function_call_outputs.push_back(output);
                            cluster_outputs.push_back(node);
                        }
                    }
                }

                // Now make a FunctionCall out of the nodes in cluster, including the new nodes
                // we just added
                auto sub_function = make_shared<Function>(cluster_outputs, cluster_inputs);
                sub_function->set_placement(placement);
                auto fc = make_shared<runtime::hybrid::op::FunctionCall>(
                    cluster_outputs, function_call_inputs, *sub_function, backend_list[placement]);
                fc->set_placement_index(0);
                for (size_t i = 0; i < cluster_outputs.size(); i++)
                {
                    // First add a GetOutputElement to the ith output of the FunctionCall
                    auto goe = make_shared<ngraph::op::GetOutputElement>(fc, i);
                    goe->set_placement_index(0);

                    auto old_source = cluster_outputs[i];
                    auto target = function_call_outputs[i];

                    std::vector<Input<Node>> target_inputs = get_inputs_from(*old_source, *target);
                    for (Input<Node> target_input : target_inputs)
                    {
                        target_input.replace_source_output(goe->output(0));
                    }
                }
            }
        }
    }
}

void runtime::hybrid::node_modifiers(const Node& node, vector<string>& attributes)
{
    vector<string> colors = {"\"#A0FFA0\"", "\"#FFF790\""};
    auto fc = dynamic_cast<const hybrid::op::FunctionCall*>(&node);
    if (fc != nullptr)
    {
        string fill_color = colors[fc->get_function()->get_placement()];
        string outline_color = colors[node.get_placement_index()];
        attributes.push_back("style=filled");
        attributes.push_back("fillcolor=" + fill_color);
        attributes.push_back("color=" + outline_color);
        attributes.push_back("penwidth=3");
    }
    else if (node.get_placement_index() < colors.size())
    {
        string color = colors[node.get_placement_index()];
        attributes.push_back("style=filled");
        attributes.push_back("fillcolor=" + color);
    }
}
