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
#include <list>
#include <unordered_set>

#include "ngraph/function.hpp"
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "ngraph/pass/dead_store_elimination.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

bool ngraph::pass::DeadStoreElimination::run_on_function(shared_ptr<ngraph::Function> func)
{
    unordered_set<const Node*> reachable_nodes;
    list<shared_ptr<Node>> result_list;
    deque<Node*> independent_nodes;
    unordered_map<const Node*, size_t> node_depencency_count;
    unordered_map<Node*, shared_ptr<Node>> node_map;

    traverse_nodes(func, [&reachable_nodes](shared_ptr<Node> node) {
        reachable_nodes.insert(
            node.get()); //TODO: [nikolayk] traverse_nodes automatically adds ALL parameters; some might not be used at all.
    });

    func->get_ops().remove_if(
        [&reachable_nodes](shared_ptr<Node> node) { return !reachable_nodes.count(node.get()); });
    func->get_ordered_ops().remove_if(
        [&reachable_nodes](shared_ptr<Node> node) { return !reachable_nodes.count(node.get()); });
    return false;
}
