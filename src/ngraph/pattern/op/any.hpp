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

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/pattern.hpp"

namespace ngraph
{
    namespace pattern
    {
        namespace op
        {
            class Any : public Pattern
            {
            public:
                Any(const std::shared_ptr<Node>& arg, Predicate predicate = nullptr) : Pattern(predicate)
                {
                    m_arguments.push_back(arg);
                    const_cast<std::multiset<Node*>&>(arg->users()).insert(this);
                }

                virtual void match_class(pattern::Matcher& matcher,
                                         std::shared_ptr<Node> graph_node) override
                {
                    if (!m_predicate || m_predicate(graph_node)) 
                    {
                        matcher.on_match_class(shared_from_this(), graph_node, true);
                    }
                    else 
                    {
                        assert(this->get_arguments().size() == 1);
                        matcher.on_match_class(this->get_arguments().at(0), graph_node, true);
                    }
                    
                }

                virtual std::string description() const override
                {
                    return "Any";
                }

            };
        }
    }
}
