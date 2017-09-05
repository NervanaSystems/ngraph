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

#include <list>
#include <map>
#include <memory>
#include <vector>

namespace ngraph
{
    class TopologicalSort;
    class Node;
    using node_ptr = std::shared_ptr<Node>;
}

class ngraph::TopologicalSort
{
public:
    TopologicalSort() {}

    void                      process(node_ptr);
    const std::vector<Node*>& get_sorted_list() const;

private:
    void promote_node(Node* n);

    std::map<size_t, std::list<Node*>> m_dependent_nodes;
    std::vector<Node*>                 m_sorted_list;
};
