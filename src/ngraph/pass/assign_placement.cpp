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

#include "ngraph/pass/assign_placement.hpp"
#include "ngraph/node.hpp"

using namespace std;
using namespace ngraph;

ngraph::pass::AssignPlacement::AssignPlacement(
    std::function<Placement(std::shared_ptr<Node>)> placement_policy)
    : m_placement_policy(placement_policy)
{
}

bool ngraph::pass::AssignPlacement::run_on_call_graph(const std::list<std::shared_ptr<Node>>& nodes)
{
    for (const std::shared_ptr<Node>& node : nodes)
    {
        run_on_node(node);
    }
    return false;
}

bool ngraph::pass::AssignPlacement::run_on_node(shared_ptr<Node> node)
{
    node->set_placement(m_placement_policy(node));
    return false;
}
