#include "graph_rewrite.hpp"
#include <algorithm>
#include <iostream>
#include <unordered_set>
#include "ngraph/log.hpp"
#include "ngraph/pattern/matcher.hpp"

bool ngraph::pass::GraphRewrite::run_matchers_on_nodes_list(std::list<std::shared_ptr<ngraph::Node>>& nodes, std::vector<std::shared_ptr<pattern::Matcher>> matchers)
{
	bool rewritten = false;
	for (auto node : nodes)
	{
		for (auto matcher : matchers)
		{
			NGRAPH_DEBUG << "Running matcher " << matcher << " on " << node << " , "
				<< node->get_name();
			if (!node->is_output() /*this restriction can be lifted when we find an use case for it*/
				&&
				matcher->match(node))
			{
				NGRAPH_DEBUG << "Matcher " << matcher << " matched " << node << " , "
					<< node->get_name();
				rewritten = true;
				matcher->process_match();
				break; //move onto the next node
			}
		}
	}
	return rewritten;
}

bool ngraph::pass::GraphRewrite::run_on_call_graph(std::list<std::shared_ptr<Node>>& nodes)
{
	return run_matchers_on_nodes_list(nodes, m_matchers);
}
