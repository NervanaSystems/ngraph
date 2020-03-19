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

#pragma once

#include <algorithm>
#include <functional>
#include <memory.h>

#include "ngraph/node.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/pattern/op/any.hpp"
#include "ngraph/pattern/op/any_of.hpp"
#include "ngraph/pattern/op/any_output.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/op/skip.hpp"

namespace ngraph
{
    namespace pass
    {
        class GraphRewrite;
    }

    namespace pattern
    {
        class Matcher;

        class NGRAPH_API MatcherState
        {
        public:
            MatcherState(Matcher*);
            bool finish(bool is_successful);
            ~MatcherState();

        protected:
            Matcher* m_matcher;
            PatternValueMap m_pattern_value_map;
            PatternValueMaps m_pattern_value_maps;
            size_t m_watermark;
            size_t m_capture_size;
            bool m_restore{true};
        };

        /// Matcher looks for node patterns in a computation graph. The patterns are described by an
        /// automaton that is described by an extended computation graph. The matcher executes
        /// by attempting to match the start node of the pattern to a computation graph value
        /// (output of a Node). In addition to determing if a match occurs, a pattern node may add
        /// graph nodes to a list of matched nodes, associate nodes with graph values, and start
        /// submatches. Submatches add match state changes to the enclosing match if the submatch
        /// succeeds; otherwise the state is reverted.
        ///
        /// The default match behavior of a pattern node with a graph nodes is that the computation
        /// graph value is added to the end of the matched value list and the match succeeds if the
        /// node/pattern types match and the input values match. In the case of a commutative node,
        /// the inputs can match in any order. If the matcher is in strict mode, the graph value
        /// element type and shape must also match.
        ///
        /// Pattern nodes that have different match behavior are in ngraph::pattern::op and have
        /// descriptions of their match behavior.
        class NGRAPH_API Matcher
        {
        public:
            using PatternMap = ngraph::pattern::PatternMap;

            // Avoid implicit string construction from nullptr.
            Matcher(const std::shared_ptr<Node> pattern_node, std::nullptr_t name) = delete;

            Matcher() {}
            Matcher(Output<Node>& pattern_node)
                : m_pattern_node{pattern_node}
            {
            }

            Matcher(Output<Node>& pattern_node, const std::string& name)
                : m_pattern_node(pattern_node)
                , m_name{name}
            {
            }

            /// \brief Constructs a Matcher object
            ///
            /// \param pattern_node is a pattern sub graph that will be matched against input graphs
            /// \param name is a string which is used for logging and disabling a matcher
            /// \param strict_mode forces a matcher to consider shapes and ET of nodes
            Matcher(const Output<Node>& pattern_node, const std::string& name, bool strict_mode)
                : m_pattern_node(pattern_node)
                , m_name(name)
                , m_strict_mode(strict_mode)
            {
            }

            // Some matches should start on a node rather than an output. These three constructors
            // are transition until we work out the right way to do that.
            Matcher(std::shared_ptr<Node> pattern_node);
            Matcher(std::shared_ptr<Node> pattern_node, const std::string& name);
            Matcher(std::shared_ptr<Node> pattern_node, const std::string& name, bool strict_mode);

            virtual ~Matcher() {}
            /// \brief Matches a pattern to \p graph_node
            ///
            /// \param graph_value is an input graph to be matched against
            bool match(const Output<Node>& graph_value);

            bool match(std::shared_ptr<Node> graph_node);

            /// \brief Matches a pattern to \p graph_node
            ///
            /// \param graph_value is an input graph to be matched against
            /// \param previous_matches contains previous mappings from labels to nodes to use
            bool match(const Output<Node>& graph_value, const PatternMap& previous_matches);
            bool match(const Output<Node>& graph_value, const PatternValueMap& previous_matches);

            template <typename T>
            static std::shared_ptr<T> unique_match(std::shared_ptr<Node> node)
            {
                std::shared_ptr<T> matched;
                for (auto arg : node->get_arguments())
                {
                    if (auto t_casted = as_type_ptr<T>(arg))
                    {
                        if (matched)
                        {
                            throw ngraph_error("There's more than two arguments of the same type");
                        }
                        else
                        {
                            matched = t_casted;
                        }
                    }
                }
                return matched;
            }

            bool is_contained_match(const NodeVector& exclusions = {}, bool ignore_unused = true);
            const NodeVector get_matched_nodes() { return as_node_vector(m_matched_list); }
            const OutputVector& get_matched_values() const { return m_matched_list; }
            OutputVector& get_matched_values() { return m_matched_list; }
            void reset() {}
            const std::string& get_name() { return m_name; }
            std::shared_ptr<Node> get_pattern() { return m_pattern_node.as_single_output_node(); }
            Output<Node> get_pattern_value() { return m_pattern_node; }
            std::shared_ptr<Node> get_match_root();
            Output<Node> get_match_value();
            PatternMap get_pattern_map() const;
            PatternValueMap& get_pattern_value_map() { return m_pattern_map; }
            PatternValueMaps& get_pattern_value_maps() { return m_pattern_value_maps; }
            /// \brief Low-level helper to match recurring patterns
            ///
            /// \param graph is a graph to be matched against
            /// \param pattern is a recurring pattern
            /// \param rpattern specifies a node to recur from next
            /// \param patterns a map from labels to matches

            size_t add_node(Output<Node> node);

            bool virtual match_value(const ngraph::Output<Node>& pattern_value,
                                     const ngraph::Output<Node>& graph_value);

            bool is_strict_mode() { return m_strict_mode; }
            virtual bool match_arguments(Node* pattern_node,
                                         const std::shared_ptr<Node>& graph_node);

            void capture(const std::set<Node*>& static_nodes);

            size_t get_number_of_recurrent_matches() const { return m_pattern_value_maps.size(); }
            NodeVector get_bound_nodes_for_pattern(const Output<Node>& pattern) const;
            size_t get_number_of_bound_labels() const;
            /// \brief Try a match
            MatcherState start_match();

            Output<Node> m_match_root;
            Output<Node> m_pattern_node;
            PatternValueMap m_pattern_map;
            PatternValueMaps m_pattern_value_maps;
            OutputVector m_matched_list;

        protected:
            bool match_permutation(const OutputVector& pattern_args, const OutputVector& args);

            std::string m_name{"unnamed"};
            bool m_strict_mode{false};
        };

        class NGRAPH_API RecurrentMatcher
        {
        public:
            /// \brief Constructs a RecurrentMatcher object. Reccurent Matchers are used to match
            ///        repeating patterns (e.g. RNN, LSTM, GRU cells)
            ///
            /// \param initial_pattern is a pattern sub graph describing the initial cell
            /// \param pattern is a pattern sub graph describing an individual cell
            /// \param rpattern is a (recurring) label to denote which node the next match should
            ///                 start at
            /// \param correlated_patterns is a set of labels whose bound nodes must remain the same
            ///                            across all cells
            RecurrentMatcher(const Output<Node>& initial_pattern,
                             const Output<Node>& pattern,
                             const std::shared_ptr<Node>& rpattern,
                             const std::set<std::shared_ptr<Node>>& correlated_patterns)
                : m_initial_pattern(initial_pattern)
                , m_pattern(pattern)
                , m_recurrent_pattern(rpattern)
                , m_correlated_patterns(correlated_patterns)
            {
            }

            /// \brief Constructs a RecurrentMatcher object. Reccurent Matchers are used to match
            ///        repeating patterns (e.g. RNN, LSTM, GRU cells)
            ///
            /// \param pattern is a pattern sub graph describing an individual cell
            /// \param rpattern is a (recurring) label to denote which node the next match should
            ///                 start at
            /// \param correlated_patterns is a set of labels whose bound nodes must remain the same
            ///                            across all cells
            RecurrentMatcher(const Output<Node>& pattern,
                             const std::shared_ptr<Node>& rpattern,
                             const std::set<std::shared_ptr<Node>>& correlated_patterns)
                : RecurrentMatcher(pattern, pattern, rpattern, correlated_patterns)
            {
            }

            RecurrentMatcher(const Output<Node>& initial_pattern,
                             const Output<Node>& pattern,
                             const std::shared_ptr<Node>& rpattern,
                             const std::set<std::shared_ptr<op::Label>>& correlated_patterns);

            RecurrentMatcher(const Output<Node>& pattern,
                             const std::shared_ptr<Node>& rpattern,
                             const std::set<std::shared_ptr<op::Label>>& correlated_patterns)
                : RecurrentMatcher(pattern, pattern, rpattern, correlated_patterns)
            {
            }

            /// \brief Returns a vector of bound nodes for a given label (used in a pattern
            /// describing an individual cell
            NodeVector get_bound_nodes_for_pattern(const std::shared_ptr<Node>& pattern) const
            {
                if (m_matches.count(pattern) == 0)
                {
                    throw ngraph_error("No bound nodes for a given label");
                }

                return as_node_vector(m_matches.at(pattern));
            }

            size_t get_number_of_recurrent_matches() const
            {
                if (m_matches.size() == 0)
                {
                    return 0;
                }

                return (*m_matches.begin()).second.size();
            }

            size_t get_number_of_bound_labels() const { return m_matches.size(); }
            /// \brief Tries to match a pattern for an individual cell to a given \p graph
            bool match(Output<Node> graph);

            std::shared_ptr<Node> get_match_root() { return m_match_root.get_node_shared_ptr(); }
            Output<Node> get_match_value() { return m_match_root; }
        private:
            Output<Node> m_initial_pattern;
            Output<Node> m_pattern;
            std::shared_ptr<Node> m_recurrent_pattern;
            const std::set<std::shared_ptr<Node>> m_correlated_patterns;
            RPatternValueMap m_matches;
            Output<Node> m_match_root;
        };
    }
}
