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

#include <algorithm>
#include <regex>

#include "matcher.hpp"
#include "ngraph/env_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/parameter.hpp"

using namespace std;
using namespace ngraph;

pattern::MatcherState::MatcherState(Matcher* matcher)
    : m_matcher(matcher)
    , m_pattern_value_map(matcher->m_pattern_map)
    , m_watermark(matcher->m_matched_list.size())
    , m_capture_size(matcher->m_pattern_value_maps.size())
{
}

pattern::Matcher::Matcher() {}

pattern::Matcher::Matcher(Output<Node>& pattern_node)
    : m_pattern_node{pattern_node}
{
}

pattern::Matcher::Matcher(Output<Node>& pattern_node, const std::string& name)
    : m_pattern_node(pattern_node)
    , m_name{name}
{
}

pattern::Matcher::Matcher(const Output<Node>& pattern_node,
                          const std::string& name,
                          bool strict_mode)
    : m_pattern_node(pattern_node)
    , m_name(name)
    , m_strict_mode(strict_mode)
{
}

pattern::Matcher::Matcher(shared_ptr<Node> pattern_node)
    : m_pattern_node(pattern_node->output(0))
{
}

pattern::Matcher::Matcher(shared_ptr<Node> pattern_node, const string& name)
    : m_pattern_node(pattern_node->output(0))
    , m_name(name)
{
}

pattern::Matcher::Matcher(shared_ptr<Node> pattern_node, const string& name, bool strict_mode)
    : Matcher(pattern_node->output(0), name, strict_mode)
{
}

pattern::MatcherState::~MatcherState()
{
    if (m_restore)
    {
        if (!m_matcher->m_matched_list.empty())
        {
            m_matcher->m_matched_list.erase(m_matcher->m_matched_list.begin() + m_watermark,
                                            m_matcher->m_matched_list.end());
        }

        if (!m_pattern_value_maps.empty())
        {
            m_matcher->m_pattern_value_maps.erase(m_pattern_value_maps.begin() + m_capture_size,
                                                  m_pattern_value_maps.end());
        }

        m_matcher->m_pattern_map = m_pattern_value_map;
    }
}

bool pattern::MatcherState::finish(bool is_successful)
{
    m_restore = !is_successful;
    return is_successful;
}

pattern::PatternMap pattern::Matcher::get_pattern_map() const
{
    return as_pattern_map(m_pattern_map);
}

size_t pattern::Matcher::add_node(Output<Node> value)
{
    size_t result = m_matched_list.size();
    m_matched_list.push_back(value);
    return result;
}

shared_ptr<Node> pattern::Matcher::get_match_root()
{
    return m_match_root.get_node_shared_ptr();
}

pattern::MatcherState pattern::Matcher::start_match()
{
    return MatcherState(this);
}

Output<Node> pattern::Matcher::get_match_value()
{
    return m_match_root;
}

void pattern::Matcher::capture(const set<Node*>& static_nodes)
{
    m_pattern_value_maps.push_back(m_pattern_map);
    m_pattern_map.clear();
    for (auto key_value : m_pattern_value_maps.back())
    {
        if (static_nodes.count(key_value.first.get()) > 0)
        {
            m_pattern_map.insert(key_value);
        }
    }
}

bool pattern::Matcher::is_contained_match(const OutputVector& exclusions, bool ignore_unused)
{
    if (exclusions.empty())
    {
        OutputVector label_exclusions;
        for (auto entry : m_pattern_map)
        {
            // leaf label
            if (entry.first->get_input_size() == 0)
            {
                label_exclusions.push_back(entry.second.get_node_shared_ptr());
            }
        }
        return get_subgraph_outputs(get_matched_values(), label_exclusions, ignore_unused).size() <
               2;
    }

    return get_subgraph_outputs(get_matched_values(), exclusions).size() < 2;
}

bool pattern::Matcher::match_value(const Output<Node>& pattern_value,
                                   const Output<Node>& graph_value)
{
    shared_ptr<Node> pattern_node = pattern_value.get_node_shared_ptr();
    shared_ptr<Node> graph_node = graph_value.get_node_shared_ptr();

    // This env var allows one to specify node name patterns to abort pattern matching
    // at particular nodes. The upshot is that one can quickly zero in on an offending
    // fusion by disabling individual fusions or optimizations that use Matcher.
    static const string node_skip_cregex = getenv_string("NGRAPH_FAIL_MATCH_AT");
    if (!node_skip_cregex.empty())
    {
        static const regex node_skip_regex(node_skip_cregex);
        if (regex_match(graph_node->get_name(), node_skip_regex))
        {
            NGRAPH_DEBUG << "[MATCHER] Aborting at " << *graph_node
                         << " due to NGRAPH_MATCHER_SKIP set to " << node_skip_cregex;
            return false;
        }
    }
    return pattern_node->match_value(this, pattern_value, graph_value);
}

bool pattern::Matcher::match_permutation(const OutputVector& pattern_args, const OutputVector& args)
{
    for (size_t i = 0; i < args.size(); i++)
    {
        if (!match_value(pattern_args.at(i), args.at(i)))
        {
            return false;
        }
    }
    return true;
}

bool pattern::Matcher::match_arguments(Node* pattern_node, const shared_ptr<Node>& graph_node)
{
    NGRAPH_DEBUG << "[MATCHER] Match arguments at " << *graph_node << " for pattern "
                 << *pattern_node;

    auto args = graph_node->input_values();
    auto pattern_args = pattern_node->input_values();

    if (args.size() != pattern_args.size())
    {
        NGRAPH_DEBUG << "[MATCHER] Aborting at " << *graph_node << " for pattern " << *pattern_node;
        return false;
    }

    if (graph_node->is_commutative())
    {
        // TODO: [nikolayk] we don't really have to use lexicographically-based perms,
        // heap's algo should be faster
        sort(begin(pattern_args),
             end(pattern_args),
             [](const Output<Node>& n1, const Output<Node>& n2) { return n1 < n2; });
        do
        {
            auto saved = start_match();
            if (match_permutation(pattern_args, args))
            {
                return saved.finish(true);
            }
        } while (next_permutation(
            begin(pattern_args),
            end(pattern_args),
            [](const Output<Node>& n1, const Output<Node>& n2) { return n1 < n2; }));
    }
    else
    {
        return match_permutation(pattern_args, args);
    }

    NGRAPH_DEBUG << "[MATCHER] Aborting at " << *graph_node << " for pattern " << *pattern_node;
    return false;
}

bool pattern::Matcher::match(const Output<Node>& graph_value)
{
    // clear our state
    m_matched_list.clear();
    return match(graph_value, PatternValueMap{});
}

bool pattern::Matcher::match(shared_ptr<Node> node)
{
    for (Output<Node> output : node->outputs())
    {
        if (this->match(output))
        {
            return true;
        }
    }
    return false;
}

bool pattern::Matcher::match(const Output<Node>& graph_value,
                             const PatternValueMap& previous_matches)
{
    // clear our state
    m_match_root.reset();
    m_pattern_map.clear();
    m_matched_list.clear();

    // insert previous matches
    m_pattern_map.insert(previous_matches.cbegin(), previous_matches.cend());
    auto saved = start_match();
    bool is_match = saved.finish(match_value(m_pattern_node, graph_value));
    if (is_match)
    {
        m_match_root = graph_value;
    }
    return is_match;
}

bool pattern::Matcher::match(const Output<Node>& graph_value, const PatternMap& previous_matches)
{
    return match(graph_value, as_pattern_value_map(previous_matches));
}

set<shared_ptr<Node>>
    pattern::RecurrentMatcher::as_node_set(const set<shared_ptr<op::Label>>& label_set)
{
    set<shared_ptr<Node>> result;
    for (auto label : label_set)
    {
        result.insert(label);
    }
    return result;
}

pattern::RecurrentMatcher::RecurrentMatcher(
    const Output<Node>& initial_pattern,
    const Output<Node>& pattern,
    const std::shared_ptr<Node>& rpattern,
    const std::set<std::shared_ptr<Node>>& correlated_patterns)
    : m_initial_pattern(initial_pattern)
    , m_pattern(pattern)
    , m_recurrent_pattern(rpattern)
    , m_correlated_patterns(correlated_patterns)
{
}

pattern::RecurrentMatcher::RecurrentMatcher(
    const Output<Node>& pattern,
    const std::shared_ptr<Node>& rpattern,
    const std::set<std::shared_ptr<Node>>& correlated_patterns)
    : RecurrentMatcher(pattern, pattern, rpattern, correlated_patterns)
{
}

pattern::RecurrentMatcher::RecurrentMatcher(
    const Output<Node>& pattern,
    const std::shared_ptr<Node>& rpattern,
    const std::set<std::shared_ptr<op::Label>>& correlated_patterns)
    : RecurrentMatcher(pattern, pattern, rpattern, correlated_patterns)
{
}

pattern::RecurrentMatcher::RecurrentMatcher(const Output<Node>& initial_pattern,
                                            const Output<Node>& pattern,
                                            const shared_ptr<Node>& rpattern,
                                            const set<shared_ptr<op::Label>>& correlated_patterns)
    : RecurrentMatcher(initial_pattern, pattern, rpattern, as_node_set(correlated_patterns))
{
}

bool pattern::RecurrentMatcher::match(std::shared_ptr<Node> graph)
{
    for (Output<Node> output : graph->outputs())
    {
        if (match(output))
        {
            return true;
        }
    }
    return false;
}

bool pattern::RecurrentMatcher::match(Output<Node> graph)
{
    bool matched = false;
    Matcher m_initial(m_initial_pattern);
    Matcher m_repeat(m_pattern);
    Matcher& m = m_initial;
    PatternValueMap previous_matches;
    m_matches.clear();
    m_match_root = graph;

    // try to match one cell (i.e. pattern)
    while (m.match(graph, previous_matches))
    {
        matched = true;
        // move to the next cell
        graph = m.get_pattern_value_map()[m_recurrent_pattern];

        // copy bound nodes for the current pattern graph into a global matches map
        for (auto cur_match : m.get_pattern_value_map())
        {
            m_matches[cur_match.first].push_back(cur_match.second);
        }

        // pre-populate the pattern map for the next cell with the bound nodes
        // from the current match. Only bound nodes whose labels are in
        // correlated_patterns are pre-populated. Skip other labels are
        // unbounded by default
        for (auto cor_pat : m_correlated_patterns)
        {
            previous_matches[cor_pat] = m.get_pattern_value_map()[cor_pat];
        }
        m = m_repeat;
    }

    if (!matched)
    {
        m_match_root.reset();
    }

    return matched;
}

/// \brief Returns a vector of bound values for a given label (used in a pattern
/// describing an individual cell
OutputVector pattern::RecurrentMatcher::get_bound_values_for_pattern(
    const std::shared_ptr<Node>& pattern) const
{
    if (m_matches.count(pattern) == 0)
    {
        throw ngraph_error("No bound nodes for a given label");
    }

    return m_matches.at(pattern);
}

size_t pattern::RecurrentMatcher::get_number_of_recurrent_matches() const
{
    if (m_matches.size() == 0)
    {
        return 0;
    }

    return (*m_matches.begin()).second.size();
}
