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

#include "label.hpp"

void ngraph::pattern::op::Label::match_class(ngraph::pattern::Matcher& matcher, std::shared_ptr<Node> graph_node)
{
    bool is_match = true;
    if (is_binded())
    {
        if (get_binded_node() != graph_node)
        {
            NGRAPH_DEBUG << "get_binded_node " << get_binded_node()->description() << " , " << get_binded_node()
                << " NOT match " << graph_node->description() << " , " << graph_node << std::endl;
            is_match = false;
        }
    }
    else 
    {
        is_match = !m_predicate || m_predicate(graph_node);
    }

    if (is_match) 
    {
        NGRAPH_DEBUG << "Binding get_binded_node " << graph_node->description() << " , " << graph_node << " , " << graph_node->get_name() << std::endl;
        m_binded = graph_node;
    }
    else 
    {
        //matcher.reset_pattern_nodes(graph_node);
        reset();
        matcher.m_match_root.reset();
        NGRAPH_DEBUG << "MATCHER IS MATCH : " << matcher.is_match() << std::endl;
    }
}