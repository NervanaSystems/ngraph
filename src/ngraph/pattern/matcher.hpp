/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <cassert>
#include <memory.h>
#include "ngraph/node.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/pattern/op/any.hpp"
#include "ngraph/pattern/op/label.hpp"

namespace ngraph
{
    namespace pass
    {
        class GraphRewrite;
    }

    namespace pattern
    {
        using gr_callback_fn = std::function<bool(class Matcher& m)>;
        using RPatternMap = std::map<std::shared_ptr<op::Label>, NodeVector>;

        namespace op
        {
            class Label;
        }

        /// \brief Matcher matches (compares) two graphs
        ///
        class Matcher
        {
        public:
            using PatternMap = std::map<std::shared_ptr<op::Label>, std::shared_ptr<Node>>;

            /// \brief Constructs a Matcher object
            ///
            /// \param pattern_node is a pattern sub graph that will be matched against input graphs
            /// \param callback is a callback function that will be called on a successful match
            Matcher(const std::shared_ptr<Node> pattern_node = nullptr,
                    gr_callback_fn callback = nullptr)
                : m_pattern_node(pattern_node)
                , m_callback(callback)
                , m_depth(0)
            {
            }
            virtual ~Matcher() {}
            /// \brief Matches a pattern to \p graph_node
            ///
            /// \param graph_node is an input graph to be matched against
            bool match(const std::shared_ptr<Node>& graph_node);

            /// \brief Matches a pattern to \p graph_node
            ///
            /// \param graph_node is an input graph to be matched against
            /// \param previous_matches contains previous mappings from labels to nodes to use
            bool match(const std::shared_ptr<Node>& graph_node, const PatternMap& previous_matches);

            template <typename T>
            static std::shared_ptr<T> unique_match(std::shared_ptr<Node> node)
            {
                std::shared_ptr<T> matched;
                for (auto arg : node->get_input_ops())
                {
                    if (auto t_casted = std::dynamic_pointer_cast<T>(arg))
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

            bool process_match(gr_callback_fn callback = nullptr);

            void reset() {}
            std::shared_ptr<Node> pattern_node() { return m_pattern_node; }
            std::shared_ptr<Node> match_root();
            PatternMap get_pattern_map() { return PatternMap{m_pattern_map}; }
            /// \brief Low-level helper to match recurring patterns
            ///
            /// \param graph is a graph to be matched against
            /// \param pattern is a recurring pattern
            /// \param rpattern specifies a node to recur from next
            /// \param patterns a map from labels to matches
            /// \param correlated_patterns specify labels whose bound nodes should be
            /// the same across all cells
            static bool match_recurring_pattern(
                std::shared_ptr<Node> graph,
                std::shared_ptr<Node> pattern,
                std::shared_ptr<op::Label> rpattern,
                RPatternMap& patterns,
                const std::set<std::shared_ptr<op::Label>>& correlated_patterns);
            friend op::Label; //TODO: refine to match_class

        protected:
            bool virtual match_node(const std::shared_ptr<Node>& pattern_node,
                                    const std::shared_ptr<Node>& graph_node,
                                    PatternMap& pattern_map);

            virtual bool match_arguments(const std::shared_ptr<Node>& pattern_node,
                                         const std::shared_ptr<Node>& graph_node,
                                         PatternMap& pattern_map);

            std::shared_ptr<Node> m_match_root;
            std::shared_ptr<Node> m_pattern_node;
            PatternMap m_pattern_map;

        private:
            static std::string pad(size_t num) { return std::string(num, ' '); }
            bool match_permutation(const NodeVector& pattern_args,
                                   const NodeVector& args,
                                   PatternMap& pattern_map);
            bool match_pattern(const std::shared_ptr<op::Label>& pattern_node,
                               const std::shared_ptr<Node>& graph_node,
                               PatternMap& pattern_map);
            bool match_any(const std::shared_ptr<op::Any>& pattern_node,
                           const std::shared_ptr<Node>& graph_node,
                           PatternMap& pattern_map);

            gr_callback_fn m_callback;
            size_t m_depth;
        };
    }
}
