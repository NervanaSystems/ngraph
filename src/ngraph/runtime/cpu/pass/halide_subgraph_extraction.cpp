//*****************************************************************************
// Copyright 2018 Intel Corporation
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

#include <iostream>

#include <list>
#include <typeindex>
#include <typeinfo>
#include <unordered_set>

#include "ngraph/op/add.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/relu.hpp"

#include "ngraph/runtime/cpu/op/halide_op.hpp"
#include "ngraph/runtime/cpu/pass/halide_subgraph_extraction.hpp"

using namespace std;
using namespace ngraph;

#define TI(x) type_index(typeid(x))

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace halide
            {
                static const std::unordered_set<std::type_index> whitelist{
                    TI(ngraph::op::Add), TI(ngraph::op::Multiply), TI(ngraph::op::Relu)};
                static const std::unordered_set<std::type_index> skiplist{TI(ngraph::op::Parameter),
                                                                          TI(ngraph::op::Result)};
            }
        }
    }
}

// Support for multiple results, multiple outputs and getoutputelement, and multiple subgraphs in a single
// pipeline is not implemented since this should go away in favor of the "hybrid" transformer approach of
// carving out subgraphs in core ngraph

bool runtime::cpu::pass::HalideSubgraphExtraction::run_on_function(
    std::shared_ptr<ngraph::Function> function)
{
    list<shared_ptr<Node>> worklist;
    auto results = function->get_results();

    // Artificial limitation
    if (results.size() > 1)
    {
        return false;
    }

    if (function->get_result()->get_element_type() != element::f32)
    {
        return false;
    }

    for (const auto& result : results)
    {
        worklist.emplace_back(result);
    }

    unordered_set<shared_ptr<Node>> ops;
    list<shared_ptr<Node>> ordered_ops;

    while (!worklist.empty())
    {
        const auto& node = worklist.front();

        if (!halide::skiplist.count(TI(*node)))
        {
            if (halide::whitelist.count(TI(*node)))
            {
                ops.emplace(node);
                ordered_ops.emplace_back(node);
            }
            else
            {
                break;
            }
        }
        const auto& args = node->get_arguments();
        for (const auto& arg : args)
        {
            worklist.emplace_back(arg);
        }
        worklist.pop_front();
    }

    NodeVector liveins;
    for (const auto& op : ops)
    {
        const auto& args = op->get_arguments();
        for (const auto& arg : args)
        {
            if (!ops.count(arg))
            {
                liveins.emplace_back(arg);
            }
        }
    }
    ordered_ops.reverse();
    auto subgraph = make_shared<cpu::op::HalideOp>(liveins,
                                                   ordered_ops,
                                                   function->get_result()->get_element_type(),
                                                   function->get_result()->get_shape());

    replace_node(function->get_result()->get_argument(0), subgraph);
    return true;
}
