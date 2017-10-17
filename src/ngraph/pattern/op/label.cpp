// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
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

    //matcher.on_match_class(shared_from_this(),
    //    graph_node,
    //    is_match && (!m_predicate || m_predicate(graph_node)));

    //we should be good to bind now

    if (is_match) 
    {
        m_binded = graph_node;
    }
    else 
    {
        matcher.reset_pattern_nodes(graph_node);
        matcher.m_match_root.reset();
        NGRAPH_DEBUG << "MATCHER IS MATCH : " << matcher.is_match() << std::endl;
    }
}