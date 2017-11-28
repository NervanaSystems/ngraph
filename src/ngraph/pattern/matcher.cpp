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
#include "ngraph/pattern/op/any.hpp"
#include "ngraph/pattern/op/label.hpp"

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

        void Matcher::reset_pattern_nodes(
            std::shared_ptr<Node> node) //TODO: [nikolayk] this doesn't have to be recursive
        //even better we should walk the entire pattern subgraph once
        //and keep track of all pattern nodes
        {
            auto label = std::dynamic_pointer_cast<::ngraph::pattern::op::Label>(node);
            NGRAPH_DEBUG << "reset_pattern_nodes : node = " << node->get_name() << " , " << node;
            if (label)
            {
                NGRAPH_DEBUG << "reset_pattern_nodes : label = " << node->get_name() << " , "
                             << node;
                label->reset();
            }

            for (auto arg : get_arguments(node))
            {
                reset_pattern_nodes(arg);
            }
        }

        void Matcher::match_pattern(const std::shared_ptr<Node>& pattern_node,
                                    const std::shared_ptr<Node>& graph_node)
        {
            auto label = std::dynamic_pointer_cast<op::Label>(pattern_node);
            bool is_match = true;
            if (label->is_bound())
            {
                if (label->get_bound_node() != graph_node)
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
                label->bind(graph_node);
            }
            else
            {
                reset();
                m_match_root.reset();
                NGRAPH_DEBUG << "MATCHER IS MATCH : " << this->is_match();
            }
        }

        void Matcher::match_any(const std::shared_ptr<Node>& pattern_node,
                                const std::shared_ptr<Node>& graph_node)
        {
            auto any = std::dynamic_pointer_cast<op::Any>(pattern_node);
            auto predicate = any->get_predicate();

            if (!predicate || any->get_predicate()(graph_node))
            {
                on_match_class(pattern_node, graph_node, true);
            }
            else
            {
                auto args = get_arguments(pattern_node);
                assert(args.size() == 1);
                on_match_class(args.at(0), graph_node, true);
            }
        }

        void Matcher::match_class(const std::shared_ptr<Node>& pattern_node,
                                  const std::shared_ptr<Node>& graph_node)
        {
            assert(pattern_node && graph_node);
            static const auto label_type = std::type_index(typeid(op::Label));
            static const auto any_type = std::type_index(typeid(op::Any));

            const auto pattern_type = std::type_index(typeid(*&*pattern_node));
            if (pattern_type == label_type)
            {
                match_pattern(pattern_node, graph_node);
                return;
            }

            if (pattern_type == any_type) //matches PatternSkipOp semantics
            {
                match_any(pattern_node, graph_node);
                return;
            }

            on_match_class(pattern_node,
                           graph_node,
                           std::type_index(typeid(*&*pattern_node)) ==
                               std::type_index(typeid(*&*graph_node)));
        }

        void Matcher::match_arguments(const Nodes& pattern_args, const Nodes& args)
        {
            m_depth++;
            for (size_t i = 0; i < args.size(); i++)
            {
                match_class(pattern_args.at(i), args.at(i));
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
                                     bool is_match)
        {
            NGRAPH_DEBUG << pad(2 * m_depth) << "[MATCHER] "
                         << "pattern = " << pattern_node << " , " << pattern_node->get_name() << " "
                         << (is_match ? " " : "NOT ") << "matched " << graph_node << " , "
                         << graph_node->get_name();
            if (!is_match)
            {
                reset_pattern_nodes(pattern_node);
                m_match_root.reset();
                return;
            }

            auto args = get_arguments(graph_node);
            auto pattern_args = get_arguments(pattern_node);

            if (args.size() != pattern_args.size())
            {
                reset_pattern_nodes(pattern_node);
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
                    reset_pattern_nodes(pattern_node);
                    m_match_root =
                        old_match_root; //previous permutation wasn't a match; reset m_match_root
                    match_arguments(pattern_args, args);
                    if (this->is_match())
                    {
                        return;
                    }
                } while (std::next_permutation(begin(pattern_args), end(pattern_args)));
            }
            else
            {
                match_arguments(pattern_args, args);
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

        bool Matcher::match(const std::shared_ptr<Node>& pattern_node,
                            const std::shared_ptr<Node>& graph_node)
        {
            if (!pattern_node || !graph_node)
            {
                NGRAPH_DEBUG << "pattern_node or graph_node are not set; matching FAILED";
                m_match_root.reset();
            }

            NGRAPH_DEBUG << "Starting match pattern = " << pattern_node << " , "
                         << pattern_node->get_name() << " , graph_node = " << graph_node << " , "
                         << graph_node->get_name();
            reset_pattern_nodes(pattern_node);
            m_match_root = graph_node;
            match_class(pattern_node, graph_node);
            //NGRAPH_DEBUG << pad(2 * m_depth) << "is_match() " << is_match();
            return is_match();
        }
    }
}
