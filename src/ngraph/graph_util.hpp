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

#pragma once

#include <list>
#include <memory>
#include <string>
#include <unordered_map>
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
                        std::function<void(std::shared_ptr<Node>)> f);
    void traverse_nodes(const Function* p, std::function<void(std::shared_ptr<Node>)> f);

    void traverse_functions(std::shared_ptr<Function> p,
                            std::function<void(std::shared_ptr<Function>)> f);

    void free_nodes(std::shared_ptr<Function>);

    void replace_node(std::shared_ptr<Node> target,
                      std::shared_ptr<Node> replacement,
                      bool replace_output = false);
    void replace_node_users_arguments(std::shared_ptr<Node> target,
                                      std::shared_ptr<Node> replacement);

    std::list<std::shared_ptr<Node>>
        topological_sort(const std::list<std::shared_ptr<Node>>& nodes);

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

        // returns true if original node is already mapped
        bool exists(std::shared_ptr<ngraph::Node> orig) const
        {
            return (m_node_map.count(orig) != 0);
        }

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
    std::shared_ptr<ngraph::Function> clone_function(std::shared_ptr<ngraph::Function> func,
                                                     NodeMap& node_map);

    // Assert that nodes in the function is colocated and return that placement
    Placement get_colocated_function_placement(std::shared_ptr<Function> func);

    // Split function to function(s) with unique placement
    std::vector<std::shared_ptr<Function>> split_function_by_placement(
        std::shared_ptr<Function> f,
        std::unordered_map<std::shared_ptr<op::Parameter>, std::shared_ptr<Node>>&
            map_parameter_to_source_node);

    // Insert parameter node between src_node and dst_node by splitting the graph
    void insert_parameter_split_between(std::shared_ptr<Node> src_node,
                                        std::shared_ptr<Node> dst_node,
                                        std::shared_ptr<op::Parameter> p_node);
}
