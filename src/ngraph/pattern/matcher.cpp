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

namespace ngraph
{
    namespace pattern
    {
        void Matcher::match_arguments(const Nodes& pattern_args, const Nodes& args)
        {
            for (size_t i = 0; i < args.size(); i++)
            {
                pattern_args.at(i)->match_class(*this, args.at(i));
                if (!m_is_match)
                {
                    return;
                }
            }
        }

        void Matcher::on_match_class(const std::shared_ptr<ngraph::Node>& pattern_node,
                                     const std::shared_ptr<ngraph::Node>& graph_node,
                                     bool is_match)
        {
            if (!is_match)
            {
                m_is_match = false;
                return;
            }

            auto args = graph_node->get_arguments();
            auto pattern_args = pattern_node->get_arguments();

            if (args.size() != pattern_args.size())
            {
                m_is_match = false;
                return;
            }

            if (graph_node->is_commutative())
            {
                auto args_copy =
                    Nodes(args); //@TODO [nikolayk] remove if there are no implicit dependencies
                do               //on the order of arguments in the rest of the compiler
                {
                    m_is_match =
                        true; //previous permutation wasn't a match; reset m_is_match back to true
                    match_arguments(pattern_args, args_copy);
                    if (m_is_match)
                    {
                        return;
                    }
                } while (std::next_permutation(begin(args_copy), end(args_copy)));
            }
            else
            {
                match_arguments(pattern_args, args);
            }
        }

        bool Matcher::match(const std::shared_ptr<Node>& pattern_node,
                            const std::shared_ptr<Node>& graph_node)
        {
            m_is_valid = true;
            pattern_node->match_class(*this, graph_node);
            return m_is_match;
        }
    }
}
