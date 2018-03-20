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

#include "matcher.hpp"
#include <algorithm>
#include <typeindex>
#include <typeinfo>

#include "ngraph/log.hpp"
#include "ngraph/op/parameter.hpp"

namespace ngraph
{
    namespace pattern
    {
        static std::vector<std::shared_ptr<Node>> get_arguments(std::shared_ptr<Node> n)
        {
            std::unordered_set<std::shared_ptr<Node>> arguments;
            for (const auto& input : n->get_inputs())
            {
                arguments.insert(input.get_output().get_node());
            }

            return std::vector<std::shared_ptr<Node>>(
                begin(arguments), end(arguments)); //vector is needed for generating permutations
        }

        std::shared_ptr<Node> Matcher::match_root() { return m_match_root; }
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

            if (is_match) //in case label was already bound this rebinds it to the same node (harmless; and the logic seems cleaner)
            {
                auto args = get_arguments(label);
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

        bool Matcher::match_any(const std::shared_ptr<op::Any>& any,
                                const std::shared_ptr<Node>& graph_node,
                                PatternMap& pattern_map)
        {
            auto predicate = any->get_predicate();

            if (!predicate || any->get_predicate()(graph_node))
            {
                return match_arguments(any, graph_node, pattern_map);
            }
            else
            {
                auto args = get_arguments(any);
                if (args.size() != 1)
                {
                    throw ngraph_error("Any can only take one argument");
                }

                return match_node(args.at(0), graph_node, pattern_map);
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

            NGRAPH_DEBUG << pad(2 * m_depth) << "[MATCHER] in match_node : "
                         << "pattern = " << pattern_node->get_name() << " matched "
                         << graph_node->get_name();

            if (auto label_node = std::dynamic_pointer_cast<op::Label>(pattern_node))
            {
                return match_pattern(label_node, graph_node, pattern_map);
            }

            if (auto any_node = std::dynamic_pointer_cast<op::Any>(
                    pattern_node)) //matches PatternSkipOp semantics
            {
                return match_any(any_node, graph_node, pattern_map);
            }

            auto p_pattern_node = pattern_node.get();
            auto p_graph_node = graph_node.get();

            if (std::type_index(typeid(*p_pattern_node)) == std::type_index(typeid(*p_graph_node)))
            {
                return match_arguments(pattern_node, graph_node, pattern_map);
            }

            return false;
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

            auto args = get_arguments(graph_node);
            auto pattern_args = get_arguments(pattern_node);

            if (args.size() != pattern_args.size())
            {
                return false;
            }

            if (graph_node->is_commutative())
            {
                std::sort(
                    begin(pattern_args),
                    end(pattern_args)); //TODO: [nikolayk] we don't really have to use lexicographically-based perms, heap's algo should be faster
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

        bool Matcher::process_match(::ngraph::pattern::gr_callback_fn callback)
        {
            gr_callback_fn cb = m_callback;
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

        static NodeVector get_users(std::shared_ptr<Node> node)
        {
            NodeVector result;

            for (size_t i = 0; i < node->get_output_size(); ++i)
            {
                for (auto input : node->get_output_inputs(i))
                {
                    result.push_back(input->get_node());
                }
            }

            return result;
        }

        bool Matcher::match(const std::shared_ptr<Node>& graph_node)
        {
            //clear our state
            m_match_root.reset();
            m_pattern_map.clear();

            if (!m_pattern_node || !graph_node)
            {
                throw "m_pattern_node or graph_node are not set!";
            }

            (void)get_users; //to supress an unused function warning

            NGRAPH_DEBUG << "[MATCHER] Starting match pattern = " << m_pattern_node->get_name()
                         << " , graph_node = " << graph_node->get_name();

            bool is_match = match_node(m_pattern_node, graph_node, m_pattern_map);
            if (is_match)
            {
                m_match_root = graph_node;
            }
            return is_match;
        }
    }
}
