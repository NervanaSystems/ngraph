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

#include "matcher.hpp"
#include <algorithm>
#include <typeindex>
#include <typeinfo>

#include "ngraph/log.hpp"
#include "ngraph/ops/parameter.hpp"

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

        std::shared_ptr<Node> Matcher::match_root()
        {
            assert(is_match());
            return m_match_root;
        }

        void Matcher::match_pattern(const std::shared_ptr<op::Label>& label,
                                    const std::shared_ptr<Node>& graph_node,
                                    PatternMap& pattern_map)
        {
            bool is_match = true;
            if (pattern_map.count(label))
            {
                if (pattern_map[label] != graph_node)
                {
                    NGRAPH_DEBUG << "get_bound_node " << label->get_bound_node()->get_name()
                                 << " , " << label->get_bound_node() << " NOT match "
                                 << graph_node->get_name() << " , " << graph_node;
                    is_match = false;
                }
            }
            else
            {
                auto predicate = label->get_predicate();
                is_match = !predicate || predicate(graph_node);
            }

            if (is_match)
            {
                NGRAPH_DEBUG << "Binding get_bound_node " << graph_node->get_name() << " , "
                             << graph_node << " , " << graph_node->get_name();
                pattern_map[label] = graph_node;
            }
            else
            {
                reset();
                m_match_root.reset();
                NGRAPH_DEBUG << "MATCHER IS MATCH : " << this->is_match();
            }
        }

        void Matcher::match_any(const std::shared_ptr<op::Any>& any,
                                const std::shared_ptr<Node>& graph_node,
                                PatternMap& pattern_map)
        {
            auto predicate = any->get_predicate();

            if (!predicate || any->get_predicate()(graph_node))
            {
                on_match_class(any, graph_node, pattern_map, true);
            }
            else
            {
                auto args = get_arguments(any);
                assert(args.size() == 1);
                on_match_class(args.at(0), graph_node, pattern_map, true);
            }
        }

        void Matcher::match_class(const std::shared_ptr<Node>& pattern_node,
                                  const std::shared_ptr<Node>& graph_node,
                                  PatternMap& pattern_map)
        {
            assert(pattern_node && graph_node);
            if (auto label_node = std::dynamic_pointer_cast<op::Label>(pattern_node))
            {
                match_pattern(label_node, graph_node, pattern_map);
                return;
            }

            if (auto any_node = std::dynamic_pointer_cast<op::Any>(
                    pattern_node)) //matches PatternSkipOp semantics
            {
                match_any(any_node, graph_node, pattern_map);
                return;
            }

            on_match_class(pattern_node,
                           graph_node,
                           pattern_map,
                           std::type_index(typeid(*&*pattern_node)) ==
                               std::type_index(typeid(*&*graph_node)));
        }

        void Matcher::match_arguments(const Nodes& pattern_args,
                                      const Nodes& args,
                                      PatternMap& pattern_map)
        {
            m_depth++;
            for (size_t i = 0; i < args.size(); i++)
            {
                match_class(pattern_args.at(i), args.at(i), pattern_map);
                if (!is_match())
                {
                    m_depth--;
                    return;
                }
            }
            m_depth--;
        }

        void Matcher::on_match_class(const std::shared_ptr<ngraph::Node>& pattern_node,
                                     const std::shared_ptr<ngraph::Node>& graph_node,
                                     PatternMap& pattern_map,
                                     bool is_match)
        {
            NGRAPH_DEBUG << pad(2 * m_depth) << "[MATCHER] "
                         << "pattern = " << pattern_node << " , " << pattern_node->get_name() << " "
                         << (is_match ? " " : "NOT ") << "matched " << graph_node << " , "
                         << graph_node->get_name();
            if (!is_match)
            {
                //reset_pattern_nodes(pattern_node);
                m_match_root.reset();
                return;
            }

            auto args = get_arguments(graph_node);
            auto pattern_args = get_arguments(pattern_node);

            if (args.size() != pattern_args.size())
            {
                //reset_pattern_nodes(pattern_node);
                m_match_root.reset();
                return;
            }

            if (graph_node->is_commutative())
            {
                auto old_match_root = m_match_root;
                std::sort(
                    begin(pattern_args),
                    end(pattern_args)); //TODO: [nikolayk] we don't really have to use lexicographically-based perms, heap's algo should be faster
                do
                {
                    NGRAPH_DEBUG << pad(2 * m_depth) << "Running a permutation for graph_node "
                                 << graph_node->get_name() << " , " << graph_node;
                    PatternMap copy{pattern_map};
                    //reset_pattern_nodes(pattern_node);
                    m_match_root =
                        old_match_root; //previous permutation wasn't a match; reset m_match_root
                    match_arguments(pattern_args, args, copy);
                    if (this->is_match())
                    {
                        pattern_map.insert(begin(copy), end(copy));
                        return;
                    }
                } while (std::next_permutation(begin(pattern_args), end(pattern_args)));
            }
            else
            {
                PatternMap copy{pattern_map};
                match_arguments(pattern_args, args, copy);
                if (this->is_match())
                {
                    pattern_map.insert(begin(copy), end(copy));
                }
            }
        }

        void Matcher::process_match(::ngraph::pattern::gr_callback_fn callback)
        {
            gr_callback_fn cb = m_callback;
            if (callback)
            {
                cb = callback;
            }

            assert(cb);
            assert(is_match());
            cb(*this);
        }

        static Nodes get_users(std::shared_ptr<Node> node)
        {
            Nodes result;

            for (auto& output : node->get_outputs())
            {
                for (auto input : output.get_inputs())
                {
                    result.push_back(input->get_node());
                }
            }

            return result;
        }

        bool Matcher::match(const std::shared_ptr<Node>& graph_node)
        {
            if (!m_pattern_node || !graph_node)
            {
                NGRAPH_DEBUG << "pattern_node or graph_node are not set; matching FAILED";
                m_match_root.reset();
            }

            if (get_users(m_pattern_node).size())
            {
                throw "Pattern Node must not be used elsewhere!";
            }

            m_pattern_map.clear();

            NGRAPH_DEBUG << "Starting match pattern = " << m_pattern_node << " , "
                         << m_pattern_node->get_name() << " , graph_node = " << graph_node << " , "
                         << graph_node->get_name();

            //reset_pattern_nodes(pattern_node);

            m_pattern_map.clear();
            m_match_root = graph_node;
            match_class(m_pattern_node, graph_node, m_pattern_map);
            return is_match();
        }
    }
}
