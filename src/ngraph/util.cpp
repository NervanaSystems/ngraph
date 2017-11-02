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

#include <deque>
#include <forward_list>
#include <iomanip>
#include <map>
#include <stack>
#include <unordered_set>

#include "ngraph/except.hpp"
#include "ngraph/function.hpp"
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "ngraph/util.hpp"

using namespace std;

map<string, ngraph::stopwatch*> ngraph::stopwatch_statistics;

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

void ngraph::traverse_postorder(std::shared_ptr<Node> n,
                                std::function<void(std::shared_ptr<Node>)> process_node,
                                std::function<bool(std::shared_ptr<Node>)> process_children)
{
    stack<shared_ptr<Node>> stack;
    stack.push(n);

    unordered_set<shared_ptr<Node>> visited;

    while (!stack.empty())
    {
        auto current = stack.top();
        if (visited.count(current))
        {
            process_node(current);
            stack.pop();
        }
        else
        {
            visited.insert(current);
            if (process_children(current))
            {
                for (auto arg : current->get_arguments())
                {
                    stack.push(arg);
                }
            }
        }
    }
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

    stack.push_front(p->get_result());
    for (auto param : p->get_parameters())
    {
        stack.push_front(param);
    }

    while (stack.size() > 0)
    {
        shared_ptr<Node> n = stack.front();
        if (instances_seen.find(n) == instances_seen.end())
        {
            instances_seen.insert(n);
            f(n);
        }
        stack.pop_front();
        for (auto arg : n->get_arguments())
        {
            stack.push_front(arg);
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

ngraph::ShapeTuple ngraph::get_shape_et(std::shared_ptr<ngraph::Node> n)
{
    auto arg_type = n->get_value_type();
    if (nullptr == arg_type)
    {
        throw ngraph::ngraph_error("Argument to sum is missing type.");
    }
    auto arg_tensor_view_type = dynamic_pointer_cast<const ngraph::TensorViewType>(arg_type);
    if (nullptr == arg_tensor_view_type)
    {
        throw ngraph::ngraph_error("Argument to sum is not a tensor view");
    }

    auto& arg_element_type = arg_tensor_view_type->get_element_type();
    auto arg_shape = arg_tensor_view_type->get_shape();
    return ngraph::ShapeTuple{arg_shape, arg_element_type};
}