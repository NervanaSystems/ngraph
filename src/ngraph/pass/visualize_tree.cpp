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

#include <fstream>

#include "ngraph/function.hpp"
#include "ngraph/node.hpp"
#include "ngraph/pass/pass.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

bool pass::VisualizeTree::run_on_function(ngraph::Function* func)
{
    // map<size_t, list<node_ptr>> dependent_nodes;
    traverse_nodes(func->get_result(), [&](Node* node) {
        for (auto arg : node->get_arguments())
        {
            m_ss << add_attributes(arg.get());
            m_ss << add_attributes(node);
            m_ss << "    " << arg->get_name() << " -> " << node->get_name();
            m_ss << ";\n";
        }
    });

    render();

    return false;
}

pass::VisualizeTree::VisualizeTree(const string& file_name)
    : m_name{file_name}
{
}

std::string pass::VisualizeTree::add_attributes(const Node* node)
{
    string rc;
    if (!contains(m_nodes_with_attributes, node))
    {
        m_nodes_with_attributes.insert(node);
        rc = get_attributes(node);
    }
    return rc;
}

std::string pass::VisualizeTree::get_attributes(const Node* node)
{
    stringstream ss;
    if (node->is_parameter())
    {
        ss << "    " << node->get_name() << " [shape=box color=blue]\n";
    }
    else
    {
        ss << "    " << node->get_name() << " [shape=ellipse color=black]\n";
    }
    return ss.str();
}

void pass::VisualizeTree::render() const
{
#ifdef GRAPHVIZ_FOUND
    auto tmp_file = m_name + ".tmp";
    ofstream out(tmp_file);
    if (out)
    {
        out << "digraph ngraph\n{\n";
        out << m_ss.str();
        out << "}\n";
        out.close();

        stringstream ss;
        ss << "dot -Tpng " << tmp_file << " -o " << m_name;
        auto cmd = ss.str();
        auto stream = popen(cmd.c_str(), "r");
        pclose(stream);

        remove(tmp_file.c_str());
    }
#endif
}
