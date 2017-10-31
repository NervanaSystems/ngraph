#include "macro.hpp"

using namespace ngraph::op;



std::shared_ptr<ngraph::Node> MacroNode::get_lowered_node() 
{
	if (!m_lowered_node)
	{
		m_lowered_node = lower();
	}

	return m_lowered_node;
}

void MacroNode::propagate_types() 
{
	auto ln = get_lowered_node();
	ln->propagate_types();
	set_value_type_checked(ln->get_value_type());
}

