#include "graph_rewrite.hpp"
#include <algorithm>

bool ngraph::pass::GraphRewrite::run_on_call_graph(std::list<Node*>&)
{
    //TODO: [nikolayk]
    //iterate over graph's nodes in a topological order
    //for each node run thru m_matcher_callback_pairs
    //if match is found trigger callback
    //callback may call replace_node
    //replace_node adds to marked_for_replacement
    //skiping over nodes marked for replacement
    //fix or re-run topological sort
    throw ngraph_error("not implemented yet.");
};

void ngraph::pass::GraphRewrite::replace_node(std::shared_ptr<Node> target,
                                              std::shared_ptr<Node> replacement)
{
    for (auto user : target->users())
    {
        auto args = user->get_arguments();
        auto it = std::find(begin(args), end(args), target);
        assert(it != end(args));
        it = args.erase(it);
        args.insert(it, replacement);
        const_cast<std::multiset<Node*>&>(replacement->users())
            .insert(user); //TODO: ask why is this set marked const?
    }

    marked_for_replacement.insert(
        target); //to make sure graph traversal skips over nodes marked for replacement

    const_cast<std::multiset<Node*>&>(target->users()).clear();

    //TODO: [nikolayk] recursively walk target and update users()
    //nodes w/ empty users sets should be DSE'ed.
}