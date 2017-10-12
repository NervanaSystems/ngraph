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
            is_match = false;
        }
    }
    //no else : we don't bind the node immediately since the subgraph underneath graph_node needs to match as well

    matcher.on_match_class(shared_from_this(),
        graph_node,
        is_match && (!m_predicate || m_predicate(graph_node)));

    //we should be good to bind now
    if (matcher.is_match())
    {
        m_binded = graph_node;
    }
}