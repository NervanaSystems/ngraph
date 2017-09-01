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

#include "node.hpp"
#include "topological_sort.hpp"
#include "util.hpp"

using namespace ngraph;
using namespace std;

void ngraph::TopologicalSort::promote_node(Node* n)
{
    for (auto dn=m_dependent_nodes.begin(); dn!=m_dependent_nodes.end(); dn++)
    {
        if (dn->first > 0)   // Skip zero as they should never be promoted
        {
            auto it = find(dn->second.begin(), dn->second.end(), n);
            if (it != dn->second.end())
            {
                // found the node
                dn->second.erase(it);
                m_dependent_nodes[dn->first-1].push_back(n);
            }
        }
    }
}

void ngraph::TopologicalSort::process(node_ptr p)
{
    traverse_nodes(p, [&](node_ptr node)
    {
        list<Node*>& node_list = m_dependent_nodes[node->arguments().size()];
        node_list.push_back(node.get());
    });

    list<Node*>& independent_nodes = m_dependent_nodes[0];
    while (independent_nodes.size() > 0)
    {
        auto independent_node = independent_nodes.front();
        m_sorted_list.push_back(independent_node);
        independent_nodes.pop_front();

        for (auto user : independent_node->users())
        {
            promote_node(user);
        }
    }
}

const std::vector<Node*>& ngraph::TopologicalSort::get_sorted_list() const
{
    return m_sorted_list;
}
