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

#pragma once

#include <deque>
#include <functional>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ngraph/function.hpp"
#include "ngraph/node.hpp"
#include "ngraph/placement.hpp"

namespace ngraph
{
    namespace descriptor
    {
        class Input;
        class Output;
    }

    namespace op
    {
        class Parameter;
    }

    void traverse_nodes(const std::shared_ptr<const Function> p,
                        std::function<void(std::shared_ptr<Node>)> f,
                        bool include_control_deps = false);
    void traverse_nodes(const Function* p,
                        std::function<void(std::shared_ptr<Node>)> f,
                        bool include_control_deps);

    void traverse_nodes(const NodeVector& io_nodes,
                        std::function<void(std::shared_ptr<Node>)> f,
                        bool include_control_deps);

    void traverse_functions(std::shared_ptr<Function> p,
                            std::function<void(std::shared_ptr<Function>)> f);

    void replace_node(std::shared_ptr<Node> target, std::shared_ptr<Node> replacement);

    template <typename T>
    std::list<std::shared_ptr<Node>> topological_sort(const T& nodes,
                                                      bool include_control_deps = false)
    {
        std::deque<ngraph::Node*> independent_nodes;
        std::unordered_map<const ngraph::Node*, size_t> node_dependency_count;
        std::unordered_map<ngraph::Node*, std::shared_ptr<ngraph::Node>> node_map;
        std::unordered_map<ngraph::Node*, std::set<Node*>> control_deps_users;

        for (auto node : nodes)
        {
            //build an equivalent of node->get_users() but for control dependencies
            size_t control_deps_count = 0;
            if (include_control_deps)
            {
                for (auto cd : node->get_control_dependencies())
                {
                    control_deps_count++;
                    control_deps_users[cd.get()].insert(node.get());
                }
            }

            node_map[node.get()] = node;
            size_t deps_count = node->get_inputs().size() + control_deps_count;
            node_dependency_count[node.get()] = deps_count;
            if (deps_count == 0)
            {
                independent_nodes.push_back(node.get());
            }
        }

        std::list<std::shared_ptr<ngraph::Node>> result_list;
        while (independent_nodes.size() > 0)
        {
            auto independent_node = independent_nodes.front();
            result_list.push_back(node_map[independent_node]);
            independent_nodes.pop_front();

            for (auto& output : independent_node->get_outputs())
            {
                for (auto& input : output.get_inputs())
                {
                    auto user = input->get_raw_pointer_node();
                    node_dependency_count[user] -= 1;
                    size_t count = node_dependency_count[user];
                    if (count == 0)
                    {
                        independent_nodes.push_back(user);
                    }
                }
            }

            if (include_control_deps)
            {
                auto cdit = control_deps_users.find(independent_node);
                if (cdit != control_deps_users.end())
                    for (auto cd_user : cdit->second)
                    {
                        node_dependency_count[cd_user] -= 1;
                        size_t count = node_dependency_count[cd_user];
                        if (count == 0)
                        {
                            independent_nodes.push_back(cd_user);
                        }
                    }
            }
        }

        NGRAPH_ASSERT(nodes.size() == result_list.size());
        return result_list;
    }

    template <typename T>
    std::list<std::shared_ptr<Node>> subgraph_topological_sort(const T& nodes,
                                                               bool include_control_deps = false)
    {
        std::deque<ngraph::Node*> independent_nodes;
        std::unordered_map<const ngraph::Node*, size_t> node_dependency_count;
        std::unordered_map<ngraph::Node*, std::shared_ptr<ngraph::Node>> node_map;
        std::unordered_map<ngraph::Node*, std::set<Node*>> control_deps_users;
        std::unordered_set<std::shared_ptr<ngraph::Node>> nodes_set(nodes.begin(), nodes.end());

        for (auto node : nodes)
        {
            //build an equivalent of node->get_users() but for control dependencies
            size_t deps_count = 0;
            if (include_control_deps)
            {
                for (auto cd : node->get_control_dependencies())
                {
                    if (nodes_set.count(cd) != 0)
                    {
                        control_deps_users[cd.get()].insert(node.get());
                        deps_count++;
                    }
                }
            }

            node_map[node.get()] = node;
            for (auto arg : node->get_arguments())
            {
                if (nodes_set.count(arg) != 0)
                {
                    deps_count++;
                }
            }

            node_dependency_count[node.get()] = deps_count;
            if (deps_count == 0)
            {
                independent_nodes.push_back(node.get());
            }
        }

        std::list<std::shared_ptr<ngraph::Node>> result_list;
        while (independent_nodes.size() > 0)
        {
            auto independent_node = independent_nodes.front();
            result_list.push_back(node_map[independent_node]);
            independent_nodes.pop_front();

            for (auto& output : independent_node->get_outputs())
            {
                for (auto& input : output.get_inputs())
                {
                    auto user = input->get_raw_pointer_node();
                    node_dependency_count[user] -= 1;
                    size_t count = node_dependency_count[user];
                    if (count == 0)
                    {
                        independent_nodes.push_back(user);
                    }
                }
            }

            if (include_control_deps)
            {
                auto cdit = control_deps_users.find(independent_node);
                if (cdit != control_deps_users.end())
                    for (auto cd_user : cdit->second)
                    {
                        node_dependency_count[cd_user] -= 1;
                        size_t count = node_dependency_count[cd_user];
                        if (count == 0)
                        {
                            independent_nodes.push_back(cd_user);
                        }
                    }
            }
        }

        NGRAPH_ASSERT(nodes.size() == result_list.size());
        return result_list;
    }

    template <typename T>
    void validate_nodes_and_infer_types(const T& nodes)
    {
        for (auto node : topological_sort(nodes))
        {
            node->delayed_validate_and_infer_types();
        }
    }

    // Check if all paths from X to a result go through Y
    bool is_post_dominated(Node* X, Node* Y);

    bool is_equal_to_const_value(std::string const_value, std::shared_ptr<Node> reduce_constant);

    // maps original to replacement nodes e.g. for clone utilities
    // performs index checking on access
    class NodeMap
    {
    public:
        // map original node to replacement node
        // throws ngraph_error if key already exists
        void add(std::shared_ptr<ngraph::Node> orig, std::shared_ptr<ngraph::Node> replacement);

        // get replacement node from original node
        // throws ngrah_error if key does not exist
        std::shared_ptr<ngraph::Node> get(std::shared_ptr<ngraph::Node> orig) const;

        template <typename T>
        T dynamic_get(const T& orig)
        {
            return std::dynamic_pointer_cast<typename T::element_type>(get(orig));
        }

        // returns true if original node is already mapped
        bool exists(std::shared_ptr<ngraph::Node> orig) const
        {
            return (m_node_map.count(orig) != 0);
        }

        void update(std::shared_ptr<ngraph::Node> orig, std::shared_ptr<ngraph::Node> val);

        const std::unordered_map<std::shared_ptr<ngraph::Node>, std::shared_ptr<ngraph::Node>>&
            get_node_map() const
        {
            return m_node_map;
        }
        std::unordered_map<std::shared_ptr<ngraph::Node>, std::shared_ptr<ngraph::Node>>&
            get_node_map()
        {
            return m_node_map;
        }

    private:
        std::unordered_map<std::shared_ptr<ngraph::Node>, std::shared_ptr<ngraph::Node>> m_node_map;
    };

    // input nodes are cloned and returned
    // NodeMap input may contain default node mapping i.e. pre-cloned nodes
    // NodeMap output (by reference) fully maps input and cloned nodes
    std::list<std::shared_ptr<ngraph::Node>>
        clone_nodes(const std::list<std::shared_ptr<ngraph::Node>>& nodes, NodeMap& node_map);

    // input function is cloned and returned
    // NodeMap input may contain default node mapping i.e. pre-cloned nodes
    // NodeMap output (by reference) fully maps input and cloned function ops
    std::shared_ptr<ngraph::Function> clone_function(const ngraph::Function& func,
                                                     NodeMap& node_map);

    // input function is cloned and returned
    std::shared_ptr<ngraph::Function> clone_function(const ngraph::Function& func);

    // Assert that nodes in the function is colocated and return that placement
    Placement get_colocated_function_placement(std::shared_ptr<Function> func);

    std::pair<std::shared_ptr<op::Result>, std::shared_ptr<op::Parameter>>
        insert_result_parameter_split(const std::shared_ptr<Node>& src_node,
                                      const std::shared_ptr<Node>& dst_node);

    void insert_new_node_between(const std::shared_ptr<Node>& src_node,
                                 const std::shared_ptr<Node>& dst_node,
                                 const std::shared_ptr<Node>& new_node);

    std::shared_ptr<Node> make_zero(const element::Type& element_type, const Shape& shape);

    std::shared_ptr<Node> make_constant_from_string(std::string val,
                                                    const element::Type& element_type,
                                                    const Shape& shape);

    bool is_zero(std::shared_ptr<Node> reduce_constant);

    NodeVector get_subgraph_outputs(const NodeVector& nodes,
                                    const NodeVector& exclusions,
                                    bool ignore_unused = false);

    bool is_one(std::shared_ptr<Node> reduce_constant);

    bool compare_constants(const std::shared_ptr<Node>& n1, const std::shared_ptr<Node>& n2);

    // Returns true if `node` is live in the graph i.e. a result op
    // transitively uses this `node`
    bool is_used(Node* node);

    // Returns count of `node` users that are still live in the graph
    size_t get_user_count(Node* node);

    // Return true if a node's user could potentially overwrite
    // the output of this node with in-place kernels
    bool possibly_overwritten(Node* node);

    bool is_strided(const Strides& strides);

    bool is_valid_rank(const std::shared_ptr<Node>& node, std::vector<size_t> valid_ranks);
}
