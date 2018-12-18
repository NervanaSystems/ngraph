//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <fstream>

#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/pass/pass.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

#define TI(x) std::type_index(typeid(x))

bool pass::VisualizeTree::run_on_module(vector<shared_ptr<ngraph::Function>>& functions)
{
    for (shared_ptr<Function> f : functions)
    {
        // map<size_t, list<node_ptr>> dependent_nodes;
        traverse_nodes(f, [&](shared_ptr<Node> node) {
            size_t i = 0;
            for (auto arg : node->get_arguments())
            {
                m_ss << add_attributes(arg);
                m_ss << add_attributes(node);
                m_ss << "    " << arg->get_name() << " -> " << node->get_name();

                if (std::getenv("NGRAPH_VISUALIZE_EDGE_LABELS") != nullptr)
                {
                    size_t output = 0;
                    if (auto goe = std::dynamic_pointer_cast<op::GetOutputElement>(node))
                    {
                        output = goe->get_n();
                    }
                    stringstream label_edge;
                    label_edge << "[label=\" " << output << " -> " << i << " \"]";
                    m_ss << label_edge.str();
                }

                m_ss << ";\n";
                i++;
            }
        });
    }

    render();

    return false;
}

pass::VisualizeTree::VisualizeTree(const string& file_name)
    : m_name{file_name}
{
}

std::string pass::VisualizeTree::add_attributes(shared_ptr<Node> node)
{
    string rc;
    if (m_nodes_with_attributes.find(node) == m_nodes_with_attributes.end())
    {
        m_nodes_with_attributes.insert(node);
        rc = get_attributes(node);
    }
    return rc;
}

std::string pass::VisualizeTree::get_attributes(shared_ptr<Node> node)
{
    stringstream ss;
    if (node->is_parameter() || node->is_output())
    {
        ss << "    " << node->get_name() << " [shape=box ";
        if (node->is_parameter())
        {
            ss << "color=blue ";
        }
        if (node->is_output())
        {
            ss << "style=filled fillcolor=pink ";
        }
    }
    else
    {
        ss << "    " << node->get_name() << " [shape=ellipse color=black";
    }

    ss << " label=\"" << node->get_name();

    static const char* nvtos = std::getenv("NGRAPH_VISUALIZE_TREE_OUTPUT_SHAPES");
    if (nvtos != nullptr)
    {
        // The shapes of the Outputs of a multi-output op
        // will be printed for its corresponding `GetOutputElement`s
        ss << " " << (node->get_outputs().size() != 1 ? std::string("[skipped]")
                                                      : vector_to_string(node->get_shape()));
    }

    static const char* nvtot = std::getenv("NGRAPH_VISUALIZE_TREE_OUTPUT_TYPES");
    if (nvtot != nullptr)
    {
        // The types of the Outputs of a multi-output op
        // will be printed for its corresponding `GetOutputElement`s
        ss << " " << ((node->get_outputs().size() != 1) ? std::string("[skipped]")
                                                        : node->get_element_type().c_type_string());
    }

    const Node& n = *node;
    auto eh = m_ops_to_details.find(TI(n));
    if (eh != m_ops_to_details.end())
    {
        eh->second(n, ss);
    }

    ss << " \"]\n";

    return ss.str();
}

std::string pass::VisualizeTree::get_file_ext()
{
    const char* format = std::getenv("NGRAPH_VISUALIZE_TREE_OUTPUT_FORMAT");
    if (!format)
    {
        format = "png";
    }

    if (format[0] == '.')
    {
        format += 1;
    }

    return std::string(format);
}

void pass::VisualizeTree::render() const
{
    auto dot_file = m_name + ".dot";
    ofstream out(dot_file);
    if (out)
    {
        out << "digraph ngraph\n{\n";
        out << m_ss.str();
        out << "}\n";
        out.close();

        stringstream ss;

        ss << "dot -T" << get_file_ext() << " " << dot_file << " -o " << m_name;
        auto cmd = ss.str();
        auto stream = popen(cmd.c_str(), "r");
        if (stream)
        {
            pclose(stream);
        }
    }
}
