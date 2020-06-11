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

static std::string label_edge(const std::shared_ptr<Node>& src,
                              const std::shared_ptr<Node>& dst,
                              size_t arg_index,
                              int64_t jump_distance)
{
    std::stringstream ss;
    if (src->get_output_size() > 1)
    {
        for (Output<Node> output : src->outputs())
        {
            for (Input<Node> input : output.get_target_inputs())
            {
                if (input.get_source_output() == output)
                {
                    stringstream label;
                    label << "[label=\" " << output.get_index() << " \"]";
                    ss << label.str();
                }
            }
        }
    }
    if (getenv_bool("NGRAPH_VISUALIZE_EDGE_LABELS"))
    {
        size_t output = 0;
        if (auto goe = as_type_ptr<op::GetOutputElement>(dst))
        {
            output = goe->get_as_output().get_index();
        }
        stringstream label;
        label << "[label=\" " << output << " -> " << arg_index << " \"]";
        ss << label.str();
    }

    else if (getenv_bool("NGRAPH_VISUALIZE_EDGE_JUMP_DISTANCE"))
    {
        if (jump_distance > 1)
        {
            stringstream label;
            label << "[label=\"jump=" << jump_distance << "\"]";
            ss << label.str();
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
        label << "label=<<table border=\"0\" cellborder=\"0\" cellpadding=\"0\" "
                 "style=\"\"><tr><td align=\"center\" colspan=\"5\">"
              << node->get_name() << "</td></tr>";

        size_t index = 0;
        const string td_start = "<td><font point-size=\"10\" face=\"courier\">";
        const string td_end = "</font></td>";
        vector<string> rows;
        vector<string> row_compare;
        for (auto input : node->inputs())
        {
            stringstream row_ss;
            stringstream row_compare_ss;
            row_ss << "<tr>";
            row_ss << td_start << "I[" << index++ << "]" << td_end;
            row_compare_ss << td_start << input.get_element_type().get_type_name() << td_end;
            row_compare_ss << td_start << pretty_partial_shape(input.get_shape()) << td_end;
            row_ss << row_compare_ss.str() << "</tr>";
            rows.push_back(row_ss.str());
            row_compare.push_back("I" + row_compare_ss.str());
        }
        index = 0;
        for (auto output : node->outputs())
        {
            stringstream row_ss;
            stringstream row_compare_ss;
            row_ss << "<tr>";
            row_ss << td_start << "O[" << index++ << "]" << td_end;
            row_compare_ss << td_start << output.get_element_type().get_type_name() << td_end;
            row_compare_ss << td_start << pretty_partial_shape(output.get_shape()) << td_end;
            row_ss << row_compare_ss.str() << "</tr>";
            rows.push_back(row_ss.str());
            row_compare.push_back("O" + row_compare_ss.str());
        }

        // Collapse duplicate rows
        vector<int64_t> remove_list;
        for (size_t i = 1; i < row_compare.size() - 1; i++)
        {
            string s1 = row_compare[i - 1];
            string s2 = row_compare[i];
            string s3 = row_compare[i + 1];
            if (s1 == s2 && s2 == s3)
            {
                remove_list.push_back(i);
            }
        }
        if (remove_list.size() > 3)
        {
            // Go backwards through the list to make removal easier
            int64_t start = remove_list[remove_list.size() - 1];
            int64_t end = start;
            int64_t count = 0;
            for (int64_t i = remove_list.size() - 2; i >= 0; --i)
            {
                int64_t row = remove_list[i];
                if (row == start - 1)
                {
                    // continue
                    start = row;
                    count++;
                }
                else
                {
                    rows.erase(rows.begin() + start, rows.begin() + end + 1);
                    string str = "<tr><td align=\"center\" colspan=\"5\">...</td></tr>";
                    rows.insert(rows.begin() + start, str);
                    end = row;
                    start = row;
                }
            }
            if (start != end)
            {
                rows.erase(rows.begin() + start, rows.begin() + end + 1);
                string str = "<tr><td align=\"center\" colspan=\"5\">...</td></tr>";
                rows.insert(rows.begin() + start, str);
            }
        }

        // if (get_provenance_enabled())
        // {
        //     for (auto tag : node->get_provenance_tags())
        //     {
        //         string str = "<tr><td align=\"left\" colspan=\"5\">tag=" + tag + "</td></tr>";
        //         rows.push_back(str);
        //     }
        // }

        for (const string& s : rows)
        {
            label << s;
        }

        label << "</table>>";
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
