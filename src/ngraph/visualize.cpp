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

#include <cstdio>
#include <fstream>
#include <list>

#include "ngraph/node.hpp"
#include "util.hpp"
#include "visualize.hpp"

using namespace ngraph;
using namespace std;

Visualize::Visualize(const string& name)
    : m_name{name}
{
}

void Visualize::add(node_ptr p)
{
    // map<size_t, list<node_ptr>> dependent_nodes;
    traverse_nodes(p, [&](node_ptr node) {
        for (auto arg : node->get_arguments())
        {
            m_ss << "    " << arg->get_node_id() << " -> " << node->get_node_id() << ";\n";
        }
    });
}

void Visualize::save_dot(const string& path) const
{
    auto     tmp_file = path + ".tmp";
    ofstream out(tmp_file);
    if (out)
    {
        out << "digraph " << m_name << "\n{\n";
        out << m_ss.str();
        out << "}\n";
        out.close();

        stringstream ss;
        ss << "dot -Tpng " << tmp_file << " -o " << path;
        auto cmd    = ss.str();
        auto stream = popen(cmd.c_str(), "r");
        pclose(stream);

        // remove(tmp_file.c_str());
    }
}
