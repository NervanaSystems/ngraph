//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "ngraph/env_util.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/function.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/experimental/compiled_kernel.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/pass/pass.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

//
// As we are visualizing the graph, we will make some tweaks to the generated dot file to make
// routing more tractable for Graphviz as well as (hopefully) more legible for the user.
//
// NOTE: It's possible, even likely, that better algorithms are available here. I just tried a
// few different things without doing much research, and this seemed to work well. Please feel
// free to improve on this. --amprocte
//
// -----------------
//
// The first tweak is to trim edges that, intuitively speaking, have long "skip distance". For
// example:
//
// [Actual Graph Structure]      [Visualization]
//    n0                             n0
//    | \                            |  \
//    n1 \                           n1  [to n50]
//    |   |                          |
//    n2  |                          n2
//    |   |                          |
//    n3  |                          n3
//    |   |                          |
//   ...  |                         ...  [from n0]
//    |  /                           |  /
//   n50                            n50
//
// This is useful for training graphs especially, which tend to have very long feed-forward edges
// for intermediate values from fprop being stored for later reuse in the bprop phase.
//
// Efficiently detecting a "long skip" is a bit tricky. We want to come up with a metric that is
// reasonably fast to compute, but does not result in cuts that will split the graph into multiple
// components. The heuristic we are using for the jump distance between n and m is the maximum
// difference in maximum path length from n and m to any result node that is reachable from both
// n and m (or 0, if no such result node exists). Not sure if this is mathematically *guaranteed*
// not to split graph components, but it seems to work well in practice.
//
// Formally:
//
// Compute-Heights-Above-Each-Parameter(N):
//    Inputs: nodes N; define R={n in N | n is a Result node}
//    Output: height_maps: map from N to (map from R to int)
//
//    height_maps is initially empty
//
//    for each r in R:
//        Insert into height_map the map {r -> 1}
//
//    for each n in N in reverse topological ("results-first") order:
//        for each user m of n:
//            for each r in height_maps[m].keys:
//                height_maps[n][r] := max(height_maps[n][r], height_maps[m][r]+1)
//
// Jump-Distance(n,m,height_maps):
//     Inputs: n (source node), m (destination node), height_maps (pre-computed above)
//     Output: jump_distance: int
//
//     jump_distance := 0
//
//     for each r in height_maps[n].keys:
//         if r is in height_maps[m].keys:
//             jump_distance := max(jump_distance, abs(height_maps[n][r] - height_maps[m][r]))
//
// Later on, if E is an edge from n to m, and Jump-Distance(n,m,height_map) > K (where K is kind
// of arbitrary but currently set to 20), we will "cut" the edge as illustrated above.
//
// -----------------
//
// The second tweak aims to eliminate routing pressure from nodes that have large outdegree and
// are connected to many otherwise-distant places in the graph. For this, the only thing we are
// doing at the moment is to "float" Parameter and Constant nodes. This means that rather than
// visualizing them as a single node (which might have very large outdegree as in, e.g., a
// learning rate parameter being fed to many different places), we make a "copy" of the node at
// each occurrence site (with a dashed outline).
//
// NOTE: This tweak could probably be extended to float other kinds of nodes with high out-degree.
// (This situation is likely to arise after constant subexpression elimination.) Here one has to
// be careful to avoid splitting the components. I have some rough ideas on how this could be
// dealt with, but have not had time to implement them yet. --amprocte
//

const int ngraph::pass::VisualizeTree::max_jump_distance = 20;

class HeightMap
{
public:
    HeightMap() {}
    HeightMap(std::set<Node*> initials)
    {
        for (auto& n : initials)
        {
            m_heights[n] = 0;
        }
    }
    void absorb(const HeightMap& other)
    {
        for (auto& p : other.m_heights)
        {
            auto k = p.first;
            auto v = p.second;
            m_heights[k] = std::max(m_heights[k], v + 1);
        }
    }
    int64_t max_jump_to(const HeightMap& target)
    {
        int64_t result = 0;
        for (auto& p : m_heights)
        {
            auto k = p.first;
            auto v = p.second;
            if (target.m_heights.count(k) != 0)
            {
                result = std::max(result, std::abs(target.m_heights.at(k) - v));
            }
        }
        return result;
    }

private:
    std::unordered_map<Node*, int64_t> m_heights;
};

static std::string label_edge(const std::shared_ptr<Node>& /* src */,
                              const std::shared_ptr<Node>& dst,
                              size_t arg_index,
                              int64_t jump_distance)
{
    std::stringstream ss;
    if (getenv_bool("NGRAPH_VISUALIZE_EDGE_LABELS"))
    {
        size_t output = 0;
        if (auto goe = as_type_ptr<op::GetOutputElement>(dst))
        {
            output = goe->get_as_output().get_index();
        }
        stringstream label_edge;
        label_edge << "[label=\" " << output << " -> " << arg_index << " \"]";
        ss << label_edge.str();
    }

    else if (getenv_bool("NGRAPH_VISUALIZE_EDGE_JUMP_DISTANCE"))
    {
        if (jump_distance > 1)
        {
            stringstream label_edge;
            label_edge << "[label=\"jump=" << jump_distance << "\"]";
            ss << label_edge.str();
        }
    }
    return ss.str();
}

bool pass::VisualizeTree::run_on_module(vector<shared_ptr<Function>>& functions)
{
    for (shared_ptr<Function> f : functions)
    {
        unordered_map<Node*, HeightMap> height_maps;

        for (auto& node : f->get_ops())
        {
            if (node->description() == "Result")
            {
                height_maps[node.get()] = HeightMap({node.get()});
            }
            else
            {
                height_maps[node.get()] = HeightMap();
            }
        }

        auto nodes = topological_sort(f->get_ops());

        for (auto it = nodes.rbegin(); it != nodes.rend(); ++it)
        {
            auto& node = *it;
            for (auto& output : node->outputs())
            {
                for (auto& input : output.get_target_inputs())
                {
                    auto target_node = input.get_node();
                    height_maps[node.get()].absorb(height_maps[target_node]);
                }
            }
        }

        // TODO(amprocte): Maybe find a way to make this tunable.

        size_t fake_node_ctr = 0;

        traverse_nodes(f, [&](shared_ptr<Node> node) {

            if (auto ck = as_type_ptr<ngraph::op::CompiledKernel>(node))
            {
                // print sub-graph
                auto nodes_list = ck->get_node_list();

                // all nodes inside the CK sub-graph
                for (auto& ck_node : nodes_list)
                {
                    m_ss << add_attributes(ck_node);
                }
                // all edges to each node in the sub-graph
                for (auto& subgraph_node : nodes_list)
                {
                    add_node_arguments(subgraph_node, height_maps, fake_node_ctr);
                }
            }
            add_node_arguments(node, height_maps, fake_node_ctr);
        });
    }

    render();

    return false;
}

pass::VisualizeTree::VisualizeTree(const string& file_name, node_modifiers_t nm, bool dot_only)
    : m_name{file_name}
    , m_node_modifiers{nm}
    , m_dot_only(dot_only)
{
}

void pass::VisualizeTree::add_node_arguments(shared_ptr<Node> node,
                                             unordered_map<Node*, HeightMap>& height_maps,
                                             size_t& fake_node_ctr)
{
    size_t arg_index = 0;
    for (auto input_value : node->input_values())
    {
        auto arg = input_value.get_node_shared_ptr();
        size_t jump_distance = height_maps[arg.get()].max_jump_to(height_maps[node.get()]);
        if (is_type<ngraph::op::Constant>(arg) || is_type<ngraph::op::Parameter>(arg))
        {
            auto clone_name = "CLONE_" + to_string(fake_node_ctr);
            auto color = (arg->description() == "Parameter" ? "blue" : "black");
            m_ss << "    " << clone_name << "[shape=\"box\" style=\"dashed,filled\" color=\""
                 << color << "\" fillcolor=\"white\" label=\"" << get_node_name(arg) << "\"]\n";
            m_ss << "    " << clone_name << " -> " << node->get_name()
                 << label_edge(arg, node, arg_index, jump_distance) << "\n";
            fake_node_ctr++;
        }
        else if (jump_distance > max_jump_distance)
        {
            m_ss << add_attributes(arg);
            m_ss << add_attributes(node);
            auto recv_node_name = "RECV_" + to_string(fake_node_ctr);
            auto send_node_name = "SEND_" + to_string(fake_node_ctr);
            m_ss << "    " << recv_node_name << "[shape=\"box\" style=\"solid,filled\" "
                                                "fillcolor=\"#ffcccc\" label=\"Receive["
                 << arg->get_name() << "]\"]\n";
            m_ss << "    " << send_node_name << "[shape=\"box\" style=\"solid,filled\" "
                                                "fillcolor=\"#ccffcc\" label=\"Send["
                 << node->get_name() << "]\"]\n";
            m_ss << "    " << arg->get_name() << " -> " << send_node_name
                 << label_edge(arg, node, arg_index, jump_distance) << "\n";
            m_ss << "    " << recv_node_name << " -> " << node->get_name()
                 << label_edge(arg, node, arg_index, jump_distance) << "\n";
            fake_node_ctr++;
        }
        else
        {
            m_ss << add_attributes(arg);
            m_ss << add_attributes(node);
            m_ss << "    " << arg->get_name() << " -> " << node->get_name()
                 << label_edge(arg, node, arg_index, jump_distance) << "\n";
        }
        arg_index++;
    }
}

string pass::VisualizeTree::add_attributes(shared_ptr<Node> node)
{
    string rc;
    if (m_nodes_with_attributes.find(node) == m_nodes_with_attributes.end())
    {
        m_nodes_with_attributes.insert(node);
        rc = get_attributes(node);
    }
    return rc;
}

static std::string pretty_partial_shape(const PartialShape& shape)
{
    std::stringstream ss;

    if (shape.rank().is_dynamic())
    {
        ss << "?";
    }
    else
    {
        bool first = true;

        ss << "[";
        for (size_t i = 0; i < shape.rank().get_length(); i++)
        {
            if (!first)
            {
                ss << ",";
            }
            if (shape[i].is_dynamic())
            {
                ss << "?";
            }
            else
            {
                ss << shape[i].get_length();
            }
            first = false;
        }
        ss << "]";
    }

    return ss.str();
}

string pass::VisualizeTree::get_attributes(shared_ptr<Node> node)
{
    vector<string> attributes;
    attributes.push_back("shape=box");

    if (node->is_output())
    {
        attributes.push_back("color=crimson");
        attributes.push_back("penwidth=1.5");
    }
    else
    {
        attributes.push_back("color=black");
    }

    // Construct the label attribute
    {
        stringstream label;
        label << "label=\"" << get_node_name(node);

        static const bool nvtos = getenv_bool("NGRAPH_VISUALIZE_TREE_OUTPUT_SHAPES");
        if (nvtos)
        {
            // The shapes of the Outputs of a multi-output op
            // will be printed for its corresponding `GetOutputElement`s
            label << " " << (node->get_output_size() != 1
                                 ? string("[skipped]")
                                 : pretty_partial_shape(node->get_output_partial_shape(0)));
        }

        static const bool nvtot = getenv_bool("NGRAPH_VISUALIZE_TREE_OUTPUT_TYPES");
        if (nvtot)
        {
            // The types of the Outputs of a multi-output op
            // will be printed for its corresponding `GetOutputElement`s
            label << " "
                  << ((node->get_output_size() != 1) ? string("[skipped]")
                                                     : node->get_element_type().c_type_string());
        }

        auto eh = m_ops_to_details.find(node->get_type_info());
        if (eh != m_ops_to_details.end())
        {
            eh->second(*node, label);
        }
        label << "\"";
        attributes.push_back(label.str());
    }

    if (m_node_modifiers)
    {
        m_node_modifiers(*node, attributes);
    }

    stringstream ss;
    ss << "    " << node->get_name() << " [" << join(attributes, " ") << "]\n";

    return ss.str();
}

string pass::VisualizeTree::get_node_name(shared_ptr<Node> node)
{
    string rc = node->get_friendly_name();
    if (node->get_friendly_name() != node->get_name())
    {
        rc += "\\n" + node->get_name();
    }
    if (auto ck = as_type_ptr<ngraph::op::CompiledKernel>(node))
    {
        rc += "\\n{";
        // add sub-graph node names
        for (auto& ck_node : ck->get_node_list())
        {
            rc += ck_node->get_name();
            rc += ", ";
        }
        rc += "}\\n";
    }
    return rc;
}

void pass::VisualizeTree::render() const
{
    string ext = file_util::get_file_ext(m_name);
    string output_format = ext.substr(1);
    string dot_file = m_name;
    if (to_lower(ext) != ".dot")
    {
        dot_file += ".dot";
    }
    ofstream out(dot_file);
    if (out)
    {
        out << "digraph ngraph\n{\n";
        out << m_ss.str();
        out << "}\n";
        out.close();

        if (!m_dot_only && to_lower(ext) != ".dot")
        {
#ifndef _WIN32
            stringstream ss;
            ss << "dot -T" << output_format << " " << dot_file << " -o" << m_name;
            auto cmd = ss.str();
            auto stream = popen(cmd.c_str(), "r");
            if (stream)
            {
                pclose(stream);
            }
#endif
        }
    }
}
