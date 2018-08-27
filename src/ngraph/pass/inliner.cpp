/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include "inliner.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/function_call.hpp"

std::vector<std::shared_ptr<ngraph::op::FunctionCall>>
    ngraph::pass::InlineSmallCalls::create_inlining_plan(std::shared_ptr<ngraph::Function> f,
                                                         size_t depth)
{
    std::vector<std::shared_ptr<ngraph::op::FunctionCall>> callees;

    for (auto n : f->get_ops())
    {
        auto fc = std::dynamic_pointer_cast<op::FunctionCall>(n);
        if (!fc)
        {
            continue;
        }
        auto callee_function = fc->get_functions().at(0);
        NGRAPH_DEBUG << "InlineSmallCalls is considering " << callee_function->get_name() << " of "
                     << fc->get_name();
        size_t callee_size = callee_function->get_ops().size();
        NGRAPH_DEBUG << "\t" << callee_function->get_name() << " size is " << callee_size
                     << " , depth = " << depth;
        if (depth < m_depth && callee_size < m_call_size_limit)
        {
            callees.push_back(fc);
        }
    }
    return callees;
}

bool ngraph::pass::Inliner::inline_function_call(std::shared_ptr<ngraph::Node> inlinee,
                                                 std::shared_ptr<ngraph::Function> caller)
{
    auto callsite = std::dynamic_pointer_cast<ngraph::op::FunctionCall>(inlinee);
    if (!callsite)
    {
        return false;
    }

    // map args to parms
    auto callee = callsite->get_functions().at(0);

    if (callee->get_results().size() > 1)
    {
        return false; // relax in the next iteration (can't just use replace_node)
    }
    ngraph::NodeMap nm;
    for (size_t i = 0; i < callee->get_parameters().size(); i++)
    {
        nm.add(callee->get_parameters().at(i), callsite->get_argument(i));
    }

    ngraph::clone_function(*callee, nm);

    auto callee_graph = nm.get(callee->get_result());
    caller->replace_node(callsite, callee_graph);
    NGRAPH_DEBUG << "Inlined " << callee->get_name() << " of " << callsite->get_name() << " into "
                 << caller->get_name();
    return true;
}

bool ngraph::pass::Inliner::run_on_function_call(std::shared_ptr<ngraph::op::FunctionCall> fc)
{
    auto f = fc->get_functions().at(0);
    NGRAPH_DEBUG << "Inliner::run_on_function on " << f->get_name();
    auto callees = m_inlining_heuristics->create_inlining_plan(f, m_depth);

    if (!callees.size())
    {
        return false;
    }

    // we could clone_function f if we need to preserve it
    run_on_functions(callees, f);
    return true;
}

void ngraph::pass::Inliner::run_on_functions(
    std::vector<std::shared_ptr<ngraph::op::FunctionCall>> callees,
    std::shared_ptr<ngraph::Function> caller)
{
    for (auto callee : callees)
    {
        m_depth++;
        // recursive inlining
        run_on_function_call(callee);
        m_depth--;
        inline_function_call(callee, caller);
    }
}

bool ngraph::pass::Inliner::run_on_module(std::vector<std::shared_ptr<ngraph::Function>>& funcs)
{
    auto outermost = funcs.front();
    NGRAPH_DEBUG << "Outermost function = " << outermost->get_name();
    auto callees = m_inlining_heuristics->create_inlining_plan(outermost, m_depth);

    if (!callees.size())
    {
        return false;
    }

    run_on_functions(callees, outermost);
    return true;
}
