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

void ngraph::pass::GraphRewrite::replace_node_io(std::shared_ptr<Node> target,
                                                 std::shared_ptr<Node> replacement)
{
    //[nikolayk] all outputs feed into all users (O by U) as per Node::assign_tensors.
    //Replacement might potentially have more/fewer outputs?
    for (auto user : get_users(target))
    {
        size_t argno =
            0; //to capture the arg number a particular user was using for target's input descs
        auto& inputs = user->get_inputs();
        inputs.erase(
            std::remove_if(begin(inputs),
                           end(inputs),
                           [target, &argno /*note a capture by ref*/](ngraph::descriptor::Input i) {
                               return i.get_node() == target;
                               argno = i.get_argno();
                           }),
            end(inputs));

        size_t index = inputs.size(); //[nikolayk] it's easier to have gaps.
        size_t arg_index = 0;
        for (auto& output : replacement->get_outputs())
        {
            inputs.emplace_back(user, index++, argno, arg_index++, output);
        }
    }
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
