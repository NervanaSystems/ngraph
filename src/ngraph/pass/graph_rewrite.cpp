#include "graph_rewrite.hpp"
#include <algorithm>
#include <iostream>
#include <unordered_set>
#include "ngraph/log.hpp"
#include "ngraph/pattern/matcher.hpp"

bool ngraph::pass::GraphRewrite::run_on_call_graph(std::list<std::shared_ptr<Node>>& nodes)
{
    bool rewritten = false;
    for (auto node : nodes)
    {
        for (auto matcher : m_matchers)
        {
            NGRAPH_DEBUG << "Running matcher " << matcher << " on " << node << " , "
                         << node->get_name();
            if (!node->is_output() && matcher->match(node))
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

static std::unordered_set<std::shared_ptr<ngraph::Node>> get_users(std::shared_ptr<ngraph::Node> n)
{
    std::unordered_set<std::shared_ptr<ngraph::Node>> users;
    for (auto& output : n->get_outputs())
    {
        for (auto& input : output.get_inputs())
        {
            users.insert(input->get_node());
        }
    }
    return users;
}

void ngraph::pass::GraphRewrite::replace_node(std::shared_ptr<Node> target,
                                                 std::shared_ptr<Node> replacement)
{
	//fix input/output descriptors
	NGRAPH_DEBUG << "Replacing target = " << target << " , " << target->get_name() << " , "
		<< "replacement = " << replacement << " , " << replacement->get_name();

	assert(target->get_outputs().size() == replacement->get_outputs().size());
	for (size_t i = 0; i < target->get_outputs().size(); i++)
	{
		auto& target_output = target->get_outputs().at(i);
		std::set<ngraph::descriptor::Input*> copy_inputs{begin(target_output.get_inputs()), end(target_output.get_inputs())}; //replace_output modifies target_output->m_inputs
		for (auto input : copy_inputs)
		{
			input->replace_output(replacement->get_outputs().at(i));
		}
	}

	//fix users and arguments
	replace_node_users_arguments(target, replacement);
}

void ngraph::pass::GraphRewrite::replace_node_users_arguments(std::shared_ptr<Node> target,
                                              std::shared_ptr<Node> replacement)
{
    NGRAPH_DEBUG << "Replacing target = " << target << " , " << target->get_name() << " , "
                 << "replacement = " << replacement << " , " << replacement->get_name();

    NGRAPH_DEBUG << "user = " << replacement << " , " << replacement->get_name();
    for (auto user : target->users())
    {
        auto& args = const_cast<ngraph::Nodes&>(user->get_arguments());
        auto it = std::find(begin(args), end(args), target);
        assert(it != end(args));
        //NGRAPH_DEBUG << "Replaced " << *it << " w/ " << replacement << " in args of " << user << " , args = " << &args;
        it = args.erase(it);
        args.insert(it, replacement);
        const_cast<std::multiset<Node*>&>(replacement->users()).insert(user);
    }
    const_cast<std::multiset<Node*>&>(target->users()).clear();

    //TODO: [nikolayk] recursively walk target and update users()
    //nodes w/ empty users sets should be DSE'ed.
}
