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

#include <cassert>
#include <deque>
#include <forward_list>
#include <iomanip>
#include <map>
#include <unordered_set>

#include "ngraph/function.hpp"
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/util.hpp"

using namespace std;

void ngraph::dump(ostream& out, const void* _data, size_t _size)
{
    auto flags = out.flags();
    const uint8_t* data = reinterpret_cast<const uint8_t*>(_data);
    size_t len = _size;
    size_t index = 0;
    while (index < len)
    {
        out << std::hex << std::setw(8) << std::setfill('0') << index;
        for (int i = 0; i < 8; i++)
        {
            if (index + i < len)
            {
                out << " " << std::hex << std::setw(2) << std::setfill('0')
                    << static_cast<uint32_t>(data[i]);
            }
            else
            {
                out << "   ";
            }
        }
        out << "  ";
        for (int i = 8; i < 16; i++)
        {
            if (index + i < len)
            {
                out << " " << std::hex << std::setw(2) << std::setfill('0')
                    << static_cast<uint32_t>(data[i]);
            }
            else
            {
                out << "   ";
            }
        }
        out << "  ";
        for (int i = 0; i < 16; i++)
        {
            char ch = (index + i < len ? data[i] : ' ');
            out << ((ch < 32) ? '.' : ch);
        }
        out << "\n";
        data += 16;
        index += 16;
    }
    out.flags(flags);
}

std::string ngraph::to_lower(const std::string& s)
{
    std::string rc = s;
    std::transform(rc.begin(), rc.end(), rc.begin(), ::tolower);
    return rc;
}

string ngraph::trim(const string& s)
{
    string rc = s;
    // trim trailing spaces
    size_t pos = rc.find_last_not_of(" \t");
    if (string::npos != pos)
    {
        rc = rc.substr(0, pos + 1);
    }

    // trim leading spaces
    pos = rc.find_first_not_of(" \t");
    if (string::npos != pos)
    {
        rc = rc.substr(pos);
    }
    return rc;
}

vector<string> ngraph::split(const string& src, char delimiter, bool do_trim)
{
    size_t pos;
    string token;
    size_t start = 0;
    vector<string> rc;
    while ((pos = src.find(delimiter, start)) != std::string::npos)
    {
        token = src.substr(start, pos - start);
        start = pos + 1;
        if (do_trim)
        {
            token = trim(token);
        }
        rc.push_back(token);
    }
    if (start <= src.size())
    {
        token = src.substr(start);
        if (do_trim)
        {
            token = trim(token);
        }
        rc.push_back(token);
    }
    return rc;
}

size_t ngraph::hash_combine(const std::vector<size_t>& list)
{
    size_t seed = 0;
    for (size_t v : list)
    {
        seed ^= v + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

void ngraph::traverse_nodes(std::shared_ptr<ngraph::Function> p,
                            std::function<void(shared_ptr<Node>)> f)
{
    traverse_nodes(p.get(), f);
}

void ngraph::traverse_nodes(ngraph::Function* p, std::function<void(shared_ptr<Node>)> f)
{
    std::unordered_set<shared_ptr<Node>> instances_seen;
    deque<shared_ptr<Node>> stack;

    for (size_t i = 0; i < p->get_output_size(); ++i)
    {
        stack.push_front(p->get_output_op(i));
    }

    for (auto param : p->get_parameters())
    {
        stack.push_front(param);
    }

    while (stack.size() > 0)
    {
        shared_ptr<Node> n = stack.front();
        if (instances_seen.count(n) == 0)
        {
            instances_seen.insert(n);
            f(n);
        }
        stack.pop_front();
        for (auto arg : n->get_input_ops())
        {
            if (instances_seen.count(arg) == 0)
            {
                stack.push_front(arg);
            }
        }
    }
}

void ngraph::traverse_functions(std::shared_ptr<ngraph::Function> p,
                                std::function<void(shared_ptr<Function>)> f)
{
    std::unordered_set<shared_ptr<Function>> instances_seen;
    deque<shared_ptr<Function>> stack;

    stack.push_front(p);

    while (stack.size() > 0)
    {
        shared_ptr<Function> func = stack.front();
        if (instances_seen.find(func) == instances_seen.end())
        {
            instances_seen.insert(func);
            f(func);
        }
        stack.pop_front();
        for (shared_ptr<Node> op : func->get_ops())
        {
            shared_ptr<Function> fp = op->get_function();
            if (fp)
            {
                stack.push_front(fp);
            }
        }
    }
}

void ngraph::free_nodes(shared_ptr<Function> p)
{
    std::deque<Node*> sorted_list;

    traverse_nodes(p, [&](shared_ptr<Node> n) { sorted_list.push_front(n.get()); });

    for (Node* n : sorted_list)
    {
        n->clear_arguments();
    }
}

void ngraph::replace_node(std::shared_ptr<Node> target, std::shared_ptr<Node> replacement)
{
    if (target->is_output()) //this restriction can be lifted when we find an use case for it
    {
        return;
    }
    //fix input/output descriptors
    NGRAPH_DEBUG << "Replacing target = " << target << " , " << target->get_name() << " , "
                 << "replacement = " << replacement << " , " << replacement->get_name();

    assert(target->get_output_size() == replacement->get_output_size());
    for (size_t i = 0; i < target->get_output_size(); i++)
    {
        std::set<ngraph::descriptor::Input*> copy_inputs{
            begin(target->get_output_inputs(i)),
            end(target->get_output_inputs(i))}; //replace_output modifies target_output->m_inputs
        for (auto input : copy_inputs)
        {
            input->replace_output(replacement, i);
        }
    }

    //fix users and arguments
    replace_node_users_arguments(target, replacement);
}

void ngraph::replace_node_users_arguments(std::shared_ptr<Node> target,
                                          std::shared_ptr<Node> replacement)
{
    NGRAPH_DEBUG << "Replacing target = " << target << " , " << target->get_name() << " , "
                 << "replacement = " << replacement << " , " << replacement->get_name();

    NGRAPH_DEBUG << "user = " << replacement << " , " << replacement->get_name();
    for (auto user : target->users())
    {
        auto& args = const_cast<ngraph::Nodes&>(user->get_arguments_FOR_GRAPH_REWRITE_ONLY());
        auto it = std::find(begin(args), end(args), target);
        assert(it != end(args));
        //NGRAPH_DEBUG << "Replaced " << *it << " w/ " << replacement << " in args of " << user << " , args = " << &args;
        it = args.erase(it);
        args.insert(it, replacement);
        const_cast<std::multiset<Node*>&>(replacement->users()).insert(user);
    }
    const_cast<std::multiset<Node*>&>(target->users()).clear();
}

std::list<std::shared_ptr<ngraph::Node>>
    ngraph::topological_sort(const std::list<std::shared_ptr<Node>>& nodes)
{
    deque<ngraph::Node*> independent_nodes;
    unordered_map<const ngraph::Node*, size_t> node_depencency_count;
    unordered_map<ngraph::Node*, shared_ptr<ngraph::Node>> node_map;

    for (auto node : nodes)
    {
        node_map[node.get()] = node;
        node_depencency_count[node.get()] = node->get_input_ops().size();
        if (node->get_input_ops().size() == 0)
        {
            independent_nodes.push_back(node.get());
        }
    }

    list<shared_ptr<ngraph::Node>> result_list;
    while (independent_nodes.size() > 0)
    {
        auto independent_node = independent_nodes.front();
        result_list.push_back(node_map[independent_node]);
        independent_nodes.pop_front();

        for (auto user : independent_node->users())
        {
            node_depencency_count[user] -= 1;
            size_t count = node_depencency_count[user];
            if (count == 0)
            {
                independent_nodes.push_back(user);
            }
        }
    }

    return result_list;
}

std::list<std::shared_ptr<ngraph::Node>>
    ngraph::clone_nodes(const std::list<std::shared_ptr<ngraph::Node>>& nodes, NodeMap& node_map)
{
    // for each node in topological order
    auto sorted_nodes = topological_sort(nodes);
    for (auto node : sorted_nodes)
    {
        if (node_map.count(node) == 0)
        {
            // get (already) cloned arguments and clone the node
            Nodes cloned_args;
            for (auto arg : node->get_input_ops())
            {
                cloned_args.push_back(node_map[arg]);
            }
            node_map[node] = node->copy_with_new_args(cloned_args);
        }
    }

    // create and return list of cloned nodes
    // order matches input list (not necessarily topological)
    std::list<std::shared_ptr<ngraph::Node>> cloned_nodes;
    for (auto node : nodes)
    {
        cloned_nodes.push_back(node_map[node]);
    }
    return cloned_nodes;
}

std::shared_ptr<ngraph::Function> ngraph::clone_function(std::shared_ptr<ngraph::Function> func,
                                                         NodeMap& node_map)
{
    // clone function operations
    clone_nodes(func->get_ops(), node_map);

    // get cloned function result and parameters
    Nodes cloned_results;
    for (size_t i = 0; i < func->get_output_size(); ++i)
    {
        cloned_results.push_back(node_map[func->get_output_op(i)]);
    }
    std::vector<std::shared_ptr<op::Parameter>> cloned_params;
    for (auto param : func->get_parameters())
    {
        cloned_params.push_back(std::dynamic_pointer_cast<op::Parameter>(node_map[param]));
    }

    // create and return cloned function
    return std::make_shared<ngraph::Function>(cloned_results, cloned_params);
}

void* ngraph::aligned_alloc(size_t alignment, size_t size)
{
#ifdef __APPLE__
    return new uint64_t[round_up(size, sizeof(uint64_t)) / sizeof(uint64_t)];
#else
    return ::aligned_alloc(alignment, size);
#endif
}

void ngraph::aligned_free(void* p)
{
#ifdef __APPLE__
    delete[] reinterpret_cast<uint64_t*>(p);
#else
    free(p);
#endif
}

size_t ngraph::round_up(size_t size, size_t alignment)
{
    if (alignment == 0)
    {
        return size;
    }

    size_t remainder = size % alignment;
    if (remainder == 0)
    {
        return size;
    }

    return size + alignment - remainder;
}

ngraph::FpropCache ngraph::cache_fprop(std::shared_ptr<ngraph::Function> fprop,
                                       std::shared_ptr<ngraph::Function> bprop,
                                       std::vector<std::shared_ptr<Node>> adjoints)
{
    using namespace ngraph;

    // Traverse fprop to make a map that stores parameters with the same
    // shape and element type as the nodes in fprop
    NodeMap node_param_map;
    ngraph::traverse_nodes(fprop, [&node_param_map](std::shared_ptr<Node> node) {
        node_param_map[node] =
            std::make_shared<op::Parameter>(node->get_element_type(), node->get_shape());
    });

    // Traverse bprop to find all of the nodes in the graph
    std::unordered_set<std::shared_ptr<Node>> in_bprop;
    ngraph::traverse_nodes(bprop, [&in_bprop](std::shared_ptr<Node> node) {
        if (in_bprop.count(node) == 0)
        {
            in_bprop.insert(node);
        }
    });

    // Get the input paramters of fprop
    std::unordered_set<std::shared_ptr<Node>> fprop_params;
    for (auto node : fprop->get_parameters())
    {
        if (fprop_params.count(node) == 0)
        {
            fprop_params.insert(node);
        }
    }

    // Find all of the nodes that are intermediate values of fprop and used in
    // bprop
    // and store those nodes that aren't needed in bprop
    FpropCache fprop_cache;
    std::vector<std::shared_ptr<Node>> unused_nodes;
    for (auto kv : node_param_map)
    {
        // if it's not in bprop, mark it unused
        if (in_bprop.count(kv.first) == 0)
        {
            unused_nodes.push_back(kv.first);
        }
        // otherwise save in in the ouputs
        else
        {
            fprop_cache.fprop_output_nodes.push_back(kv.first);
        }
    }

    // erase all unused nodes form the map
    for (auto node : unused_nodes)
    {
        node_param_map.erase(node);
    }

    // create the new outputs for fprop and the new fprop function
    Nodes fprop_outputs{fprop->get_results()};
    fprop_outputs.insert(fprop_outputs.end(),
                         fprop_cache.fprop_output_nodes.begin(),
                         fprop_cache.fprop_output_nodes.end());

    fprop_cache.fprop = std::make_shared<Function>(fprop_outputs, fprop->get_parameters());

    // clone the nodes in bprop, replacing fprop-related nodes with the
    // intermediate parameters
    ngraph::clone_nodes(bprop->get_ops(), node_param_map);

    // get cloned bprop results
    Nodes cloned_results;
    for (auto node : bprop->get_results())
    {
        cloned_results.push_back(node_param_map[node]);
    }

    // get clone bprop parameters
    op::Parameters bprop_input_params;
    for (auto param : adjoints)
    {
        bprop_input_params.push_back(
            std::dynamic_pointer_cast<op::Parameter>(node_param_map[param]));
    }

    // add the cached fprop nodes as inputs to bprop
    for (auto x : fprop_cache.fprop_output_nodes)
    {
        bprop_input_params.push_back(std::dynamic_pointer_cast<op::Parameter>(node_param_map[x]));
    }

    // create the new bprop function
    fprop_cache.bprop = std::make_shared<Function>(cloned_results, bprop_input_params);

    return fprop_cache;
}
