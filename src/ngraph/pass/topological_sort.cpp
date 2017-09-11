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

#include <deque>
#include <unordered_map>

#include "ngraph/node.hpp"
#include "ngraph/topological_sort.hpp"
#include "util.hpp"
#include "log.hpp"

using namespace ngraph;
using namespace std;

bool ngraph::pass::TopologicalSort::run_on_tree(std::shared_ptr<Node> p)
{
    deque<Node*> independent_nodes;
    unordered_map<Node*, size_t> node_depencency_count;

    traverse_nodes(p, [&](Node* node) {
        node_depencency_count[node] = node->get_arguments().size();
        if (node->get_arguments().size() == 0)
        {
            independent_nodes.push_back(node);
        }
    });

    while (independent_nodes.size() > 0)
    {
        auto independent_node = independent_nodes.front();
        m_sorted_list.push_back(independent_node);
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

    return false;
}

std::list<Node*> ngraph::pass::TopologicalSort::get_call_graph() const
{
    return m_sorted_list;
}
