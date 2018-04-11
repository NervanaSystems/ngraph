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

#include <functional>
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

    void replace_node(std::shared_ptr<Node> target, std::shared_ptr<Node> replacement);

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
}
