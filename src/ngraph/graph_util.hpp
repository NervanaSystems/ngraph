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

#pragma once

#include <algorithm>
#include <chrono>
#include <deque>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ngraph/node.hpp"

namespace ngraph
{
    class Node;
    class Function;

    template <typename T, typename V>
    void traverse_nodes(T p, V f)
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

    template <>
    void traverse_nodes(std::shared_ptr<Function> p, std::function<void(std::shared_ptr<Node>)> f);

    template <>
    void traverse_nodes(std::shared_ptr<const Function> p,
                        std::function<void(std::shared_ptr<Node>)> f);

    void traverse_functions(std::shared_ptr<Function> p,
                            std::function<void(std::shared_ptr<Function>)> f);

    void free_nodes(std::shared_ptr<Function>);

    void replace_node(std::shared_ptr<Node> target, std::shared_ptr<Node> replacement);
    void replace_node_users_arguments(std::shared_ptr<Node> target,
                                      std::shared_ptr<Node> replacement);

    std::list<std::shared_ptr<Node>>
        topological_sort(const std::list<std::shared_ptr<Node>>& nodes);

    // maps original to replacement nodes e.g. for clone utilities
    // performs index checking on access
    class NodeMap
    {
    public:
        // map original node to replcacement node
        // throws ngraph_error if key already exists
        void Add(std::shared_ptr<ngraph::Node> orig, std::shared_ptr<ngraph::Node> replacement);

        // get replacement node from original node
        // throws ngrah_error if key does not exist
        std::shared_ptr<ngraph::Node> operator[](std::shared_ptr<ngraph::Node> orig) const;

        // returns true if original node is already mapped
        bool Exists(std::shared_ptr<ngraph::Node> orig) const
        {
            return (node_map_.count(orig) != 0);
        }

    private:
        std::unordered_map<std::shared_ptr<ngraph::Node>, std::shared_ptr<ngraph::Node>> node_map_;
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
}
