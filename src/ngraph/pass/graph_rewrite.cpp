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

#include <algorithm>
#include <iostream>
#include <unordered_set>

#include "graph_rewrite.hpp"
#include "ngraph/log.hpp"
#include "ngraph/pattern/matcher.hpp"

bool ngraph::pass::GraphRewrite::run_matchers_on_nodes_list(
    const std::list<std::shared_ptr<ngraph::Node>>& nodes,
    const std::vector<std::shared_ptr<pattern::Matcher>>& matchers,
    std::shared_ptr<ngraph::Function> f)
{
    bool rewritten = false;
    for (auto node : nodes)
    {
        for (auto matcher : matchers)
        {
            NGRAPH_DEBUG << "Running matcher " << matcher->get_name() << "("
                         << matcher->get_pattern()->get_name() << ") on " << node->get_name();
            if (matcher->match(node))
            {
                NGRAPH_DEBUG << "Matcher " << matcher << matcher->get_name() << " matched "
                             << node->get_name();
                rewritten = true;
                if (matcher->process_match())
                {
                    break;
                }
            }
        }
    }
    return rewritten;
}

bool ngraph::pass::GraphRewrite::run_on_function(std::shared_ptr<ngraph::Function> f)
{
    return run_matchers_on_nodes_list(f->get_ordered_ops(), m_matchers, f);
}

bool ngraph::pass::RecurrentGraphRewrite::run_on_function(std::shared_ptr<ngraph::Function> f)
{
    bool changed = false;
    size_t i = 0;
    do
    {
        for (auto node : f->get_ops())
        {
            for (auto matcher : m_matchers)
            {
                NGRAPH_DEBUG << "Running matcher " << matcher << " on " << node->get_name();
                if (matcher->match(node))
                {
                    NGRAPH_DEBUG << "Matcher " << matcher << " matched " << node->get_name();
                    if (matcher->process_match())
                    {
                        changed = true;
                        goto next_fusion;
                    }
                }
            }
        }
    next_fusion:
        i++;
    } while (changed && i < m_num_iters);
    return changed;
}
