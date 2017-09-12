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
#include <memory>

#include "ngraph/pass/tree_pass.hpp"

namespace ngraph
{
    namespace pass
    {
        class TopologicalSort;
    }
    class Node;
}

class ngraph::pass::TopologicalSort : public TreeBase
{
public:
    TopologicalSort() {}

    bool run_on_tree(std::shared_ptr<Node>) override;

    bool             call_graph_produced() const override { return true; }
    std::list<Node*> get_call_graph() const override;

private:
    std::list<Node*> m_sorted_list;
};
