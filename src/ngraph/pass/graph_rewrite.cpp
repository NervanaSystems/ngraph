#include <algorithm>
#include "graph_rewrite.hpp"

bool ngraph::pass::GraphRewrite::run_on_call_graph(std::list<Node*>& nodes)
{
    //until @bob implements 
};

bool ngraph::pass::GraphRewrite::run_on_call_graph(std::list<Node*>& nodes) 
{
    bool rewritten = false;
    for (Node* node : nodes)
    {
        if (marked_for_replacement.find(std::make_shared<Node>(node)) == marked_for_replacement.end())
        {
            continue;
        }

        for (auto pair : m_matcher_callback_pairs)
        {
            auto matcher = pair.first;
            matcher->reset();
            if (matcher->match(std::make_shared<Node>(node)))
            {
                rewritten = true;
                pair.second(matcher, std::make_shared<Node>(node), *this);
            }
        }
    }
    return rewritten;
}

void ngraph::pass::GraphRewrite::replace_node(std::shared_ptr<Node> target, std::shared_ptr<Node> replacement) 
{
    for (auto user : target->users()) 
    {
        
        auto args = user->get_arguments();
        auto it = std::find(begin(args), end(args), target);
        assert(it != end(args));
        it = args.erase(it);
        args.insert(it, replacement);
        const_cast<std::multiset<Node*> &> (replacement->users()).insert(user); //TODO: ask why is this set marked const?
    }

    marked_for_replacement.insert(target); //to make sure graph traversal skips over nodes marked for replacement

    const_cast<std::multiset<Node*> &>(target->users()).clear();

    //TODO: [nikolayk] recursively walk target and update users() 
    //nodes w/ empty users sets should be DSE'ed.

}