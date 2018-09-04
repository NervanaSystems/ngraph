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

#include "matcher.hpp"
#include <algorithm>
#include <typeindex>
#include <typeinfo>

#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/parameter.hpp"

namespace ngraph
{
    namespace pattern
    {
        std::shared_ptr<Node> Matcher::get_match_root() { return m_match_root; }
        bool Matcher::match_pattern(const std::shared_ptr<op::Label>& label,
                                    const std::shared_ptr<Node>& graph_node,
                                    PatternMap& pattern_map)
        {
            bool is_match = true;
            if (pattern_map.count(label))
            {
                if (pattern_map[label] != graph_node)
                {
                    NGRAPH_DEBUG << "[MATCHER] get_bound_node " << pattern_map[label]->get_name()
                                 << " , " << pattern_map[label] << " does NOT match "
                                 << graph_node->get_name();
                    is_match = false;
                }
            }
            else
            {
                auto predicate = label->get_predicate();
                is_match = !predicate || predicate(graph_node);
            }

            if (is_match) // in case label was already bound this rebinds it to the same node (harmless; and the logic seems cleaner)
            {
                auto args = label->get_arguments();
                if (args.size() > 0)
                {
                    if (args.size() != 1)
                    {
                        throw ngraph_error("Labels can only take 1 argument!");
                    }
                    NGRAPH_DEBUG << "[MATCHER] Label describes a sub graph in the pattern";
                    is_match = match_node(args.at(0), graph_node, pattern_map);
                }

                if (is_match)
                {
                    NGRAPH_DEBUG << "[MATCHER] (Re)binding get_bound_node " << label->get_name()
                                 << " , " << graph_node << " , " << graph_node->get_name();
                    pattern_map[label] = graph_node;
                }
            }
            return is_match;
        }

        bool Matcher::is_contained_match(const NodeVector& exclusions, bool ignore_unused)
        {
            if (exclusions.empty())
            {
                NodeVector label_exclusions;
                for (auto entry : m_pattern_map)
                {
                    // leaf label
                    if (entry.first->get_inputs().empty())
                    {
                        label_exclusions.push_back(entry.second);
                    }
                }
                return ngraph::get_subgraph_outputs(
                           get_matched_nodes(), label_exclusions, ignore_unused)
                           .size() < 2;
            }

            return ngraph::get_subgraph_outputs(get_matched_nodes(), exclusions).size() < 2;
        }

        bool Matcher::match_skip(const std::shared_ptr<op::Skip>& skip,
                                 const std::shared_ptr<Node>& graph_node,
                                 PatternMap& pattern_map)
        {
            auto predicate = skip->get_predicate();

            if (!predicate || predicate(graph_node))
            {
                return match_arguments(skip, graph_node, pattern_map);
            }
            else
            {
                auto args = skip->get_arguments();
                if (args.size() != 1)
                {
                    throw ngraph_error("Skip can only take one argument");
                }

                return match_node(args.at(0), graph_node, pattern_map);
            }
        }

        bool Matcher::match_any(const std::shared_ptr<op::Any>& any,
                                const std::shared_ptr<Node>& graph_node,
                                PatternMap& pattern_map)
        {
            auto predicate = any->get_predicate();
            if (!predicate)
            {
                throw ngraph_error("predicate is required");
            }

            if (predicate(graph_node))
            {
                return match_arguments(any, graph_node, pattern_map);
            }
            else
            {
                return false;
            }
        }

        bool Matcher::match_node(const std::shared_ptr<Node>& pattern_node,
                                 const std::shared_ptr<Node>& graph_node,
                                 PatternMap& pattern_map)
        {
            if (!pattern_node || !graph_node)
            {
                throw ngraph_error("pattern_node or graph_node shouldn't be nullptrs!");
            }

            add_node(graph_node);
            size_t watermark = m_matched_list.size() - 1;

            NGRAPH_DEBUG << pad(2 * m_depth) << "[MATCHER] in match_node : "
                         << "pattern = " << pattern_node->get_name() << " matched "
                         << graph_node->get_name();

            if (auto label_node = std::dynamic_pointer_cast<op::Label>(pattern_node))
            {
                return abort_match(watermark, match_pattern(label_node, graph_node, pattern_map));
            }

            if (auto skip_node = std::dynamic_pointer_cast<op::Skip>(
                    pattern_node)) // matches PatternSkipOp semantics
            {
                return abort_match(watermark, match_skip(skip_node, graph_node, pattern_map));
            }

            if (auto any_node = std::dynamic_pointer_cast<op::Any>(pattern_node))
            {
                return abort_match(watermark, match_any(any_node, graph_node, pattern_map));
            }

            auto p_pattern_node = pattern_node.get();
            auto p_graph_node = graph_node.get();

            if (std::type_index(typeid(*p_pattern_node)) == std::type_index(typeid(*p_graph_node)))
            {
                return abort_match(watermark,
                                   match_arguments(pattern_node, graph_node, pattern_map));
            }

            return abort_match(watermark, false);
        }

        bool Matcher::match_permutation(const NodeVector& pattern_args,
                                        const NodeVector& args,
                                        PatternMap& pattern_map)
        {
            m_depth++;
            for (size_t i = 0; i < args.size(); i++)
            {
                if (!match_node(pattern_args.at(i), args.at(i), pattern_map))
                {
                    m_depth--;
                    return false;
                }
            }
            m_depth--;
            return true;
        }

        bool Matcher::match_arguments(const std::shared_ptr<ngraph::Node>& pattern_node,
                                      const std::shared_ptr<ngraph::Node>& graph_node,
                                      PatternMap& pattern_map)
        {
            NGRAPH_DEBUG << pad(2 * m_depth) << "[MATCHER] in match_arguments : "
                         << "pattern = " << pattern_node->get_name() << " "
                         << "matched " << graph_node->get_name();

            auto args = graph_node->get_arguments();
            auto pattern_args = pattern_node->get_arguments();

            if (args.size() != pattern_args.size())
            {
                return false;
            }

            if (graph_node->is_commutative())
            {
                std::sort(
                    begin(pattern_args),
                    end(pattern_args)); // TODO: [nikolayk] we don't really have to use lexicographically-based perms, heap's algo should be faster
                do
                {
                    NGRAPH_DEBUG << pad(2 * m_depth) << "Running a permutation for graph_node "
                                 << graph_node->get_name();
                    PatternMap copy{pattern_map};
                    if (match_permutation(pattern_args, args, copy))
                    {
                        pattern_map.insert(begin(copy), end(copy));
                        return true;
                    }
                } while (std::next_permutation(begin(pattern_args), end(pattern_args)));
            }
            else
            {
                PatternMap copy{pattern_map};
                if (match_permutation(pattern_args, args, copy))
                {
                    pattern_map.insert(begin(copy), end(copy));
                    return true;
                }
            }
            return false;
        }

        bool Matcher::process_match(::ngraph::pattern::graph_rewrite_callback callback)
        {
            graph_rewrite_callback cb = m_callback;
            if (callback)
            {
                cb = callback;
            }
            if (!cb)
            {
                throw ngraph_error("process_match invoked w/o a callback function");
            }

            if (!this->m_match_root)
            {
                throw ngraph_error("process_match invoked w/o a match");
            }

            return cb(*this);
        }

        bool Matcher::match(const std::shared_ptr<Node>& graph_node)
        {
            // clear our state
            m_match_root.reset();
            m_pattern_map.clear();
            m_matched_list.clear();

            if (!m_pattern_node || !graph_node)
            {
                throw ngraph_error("m_pattern_node or graph_node are not set");
            }

            NGRAPH_DEBUG << "[MATCHER] Starting match pattern = " << m_pattern_node->get_name()
                         << " , graph_node = " << graph_node->get_name();

            bool is_match = match_node(m_pattern_node, graph_node, m_pattern_map);
            if (is_match)
            {
                m_match_root = graph_node;
            }
            return is_match;
        }

        bool Matcher::match(const std::shared_ptr<Node>& graph_node,
                            const PatternMap& previous_matches)
        {
            // clear our state
            m_match_root.reset();
            m_pattern_map.clear();

            // insert previous matches
            m_pattern_map.insert(previous_matches.cbegin(), previous_matches.cend());

            if (!m_pattern_node || !graph_node)
            {
                throw ngraph_error("m_pattern_node or graph_node are not set");
            }

            NGRAPH_DEBUG << "[MATCHER] Starting match pattern = " << m_pattern_node->get_name()
                         << " , graph_node = " << graph_node->get_name();

            bool is_match = match_node(m_pattern_node, graph_node, m_pattern_map);
            if (is_match)
            {
                m_match_root = graph_node;
            }
            return is_match;
        }

        bool RecurrentMatcher::match(std::shared_ptr<Node> graph)
        {
            bool matched = false;
            Matcher m(m_pattern);
            Matcher::PatternMap previous_matches;
            m_matches.clear();
            m_match_root = graph;

            NGRAPH_DEBUG << "matching graph to " << graph->get_name() << std::endl;
            // try to match one cell (i.e. pattern)
            while (m.match(graph, previous_matches))
            {
                matched = true;
                // move to the next cell
                graph = m.get_pattern_map()[m_recurrent_pattern];
                NGRAPH_DEBUG << "setting graph to " << graph->get_name() << std::endl;

                // copy bound nodes for the current pattern graph into a global matches map
                for (auto cur_match : m.get_pattern_map())
                {
                    m_matches[cur_match.first].push_back(cur_match.second);
                }

                // pre-populate the pattern map for the next cell with the bound nodes
                // from the current match. Only bound nodes whose labels are in
                // correlated_patterns are pre-populated. Skip other labels are
                // unbounded by default
                for (auto cor_pat : m_correlated_patterns)
                {
                    if (m.get_pattern_map().count(cor_pat) != 0)
                    {
                        // assert that bound nodes from the previous and current matches are the same
                        if (previous_matches.count(cor_pat) != 0)
                        {
                            if (previous_matches[cor_pat] != m.get_pattern_map()[cor_pat])
                            {
                                throw ngraph_error(
                                    "previous matches and current matches aren't consistent!");
                            }
                        }

                        previous_matches[cor_pat] = m.get_pattern_map()[cor_pat];
                    }
                }
            }

            if (!matched)
            {
                m_match_root.reset();
            }

            return matched;
        }

        bool RecurrentMatcher::process_match() { return m_callback(*this); }
    }
}
