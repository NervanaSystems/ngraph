#include "graph_rewrite.hpp"
#include <algorithm>
#include <iostream>
#include "ngraph/log.hpp"
#include "ngraph/pattern/matcher.hpp"

bool ngraph::pass::GraphRewrite::run_on_call_graph(std::list<std::shared_ptr<Node>>& nodes)
{
    bool rewritten = false;
    for (auto node : nodes)
    {
        //NGRAPH_DEBUG << "Processing " << node;
        for (auto matcher : m_matchers)
        {
            NGRAPH_DEBUG << "Running matcher " << matcher << " on " << node << " , "
                         << node->description();
            if (matcher->match(node))
            {
                NGRAPH_DEBUG << "Matcher " << matcher << " matched " << node << " , "
                             << node->description();
                rewritten = true;
                matcher->process_match();
            }
        }
    }
    return rewritten;
}

void ngraph::pass::GraphRewrite::replace_node(std::shared_ptr<Node> target,
                                              std::shared_ptr<Node> replacement)
{
    NGRAPH_DEBUG << "Replacing target = " << target << " , " << target->description() << " , "
                 << "replacement = " << replacement << " , " << replacement->description();

    NGRAPH_DEBUG << "user = " << replacement << " , " << replacement->description();
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
