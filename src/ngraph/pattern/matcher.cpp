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
#include "ngraph/ngraph.hpp"
#include "ngraph/log.hpp"
#include "ngraph/pattern/op/label.hpp"

namespace ngraph
{
    namespace pattern
    {
        void Matcher::reset_pattern_nodes(std::shared_ptr<Node> node) //TODO: [nikolayk] this doesn't have to be recursive
                                                                      //even better we should walk the entire pattern subgraph once 
                                                                      //and keep track of all pattern nodes
        {
            auto label = std::dynamic_pointer_cast<::ngraph::pattern::op::Label>(node);
            NGRAPH_DEBUG << "reset_pattern_nodes : node = " << node->description() << " , " << node << std::endl;
            if (label)
            {
                NGRAPH_DEBUG << "reset_pattern_nodes : label = " << node->description() << " , " << node << std::endl;
                label->reset();
            }

            for (auto arg : node->get_arguments()) 
            {
                reset_pattern_nodes(arg);
            }
        }

        void Matcher::match_arguments(const Nodes& pattern_args, const Nodes& args)
        {
            m_depth++;
            for (size_t i = 0; i < args.size(); i++)
            {
                pattern_args.at(i)->match_class(*this, args.at(i));
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
            NGRAPH_DEBUG << pad(2*m_depth) << "[MATCHER] " << "pattern = " << pattern_node << " , " << pattern_node->description()
               << " " << (is_match ? " " : "NOT ") << "matched " << graph_node << " , " << graph_node->description();
            if (!is_match)
            {
                reset_pattern_nodes(pattern_node);
                m_match_root.reset();
                return;
            }

            auto args = graph_node->get_arguments();
            auto pattern_args = pattern_node->get_arguments();

            if (args.size() != pattern_args.size())
            {
                reset_pattern_nodes(pattern_node);
                m_match_root.reset();
                return;
            }

            
            if (graph_node->is_commutative())
            {
                auto old_match_root = m_match_root;
                std::sort(begin(pattern_args), end(pattern_args)); //TODO: [nikolayk] we don't really have to use lexicographically-based perms, heap's algo should be faster 
                do
                {
                    NGRAPH_DEBUG << pad(2 * m_depth) << "Running a permutation for graph_node " << graph_node->description() << " , " << graph_node << std::endl;
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
            NGRAPH_DEBUG << "Starting match pattern = " << pattern_node << " , " << pattern_node->description()
                << " , graph_node = " << graph_node << " , " << graph_node->description() << std::endl;
            reset_pattern_nodes(pattern_node);
            m_match_root = graph_node;
            pattern_node->match_class(*this, graph_node);
            //NGRAPH_DEBUG << pad(2 * m_depth) << "is_match() " << is_match() << std::endl;
            return is_match();
        }

    }
}
