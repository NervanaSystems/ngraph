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

#include <functional>

#include "ngraph/node.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pattern/matcher.hpp"

namespace ngraph
{
    namespace pattern
    {
        namespace op
        {
            using Predicate = std::function<bool(std::shared_ptr<Node>)>;

            class Pattern : public Node
            {
            public:
                Pattern()
                    : Pattern(nullptr){};

                Pattern(Predicate pred)
                    : Node()
                    , m_predicate(pred)
                {
                }

                virtual std::shared_ptr<Node> copy_with_new_args(
                    const std::vector<std::shared_ptr<Node>>& new_args) const override
                {
                    if (new_args.size() != 0)
                        throw ngraph_error("Incorrect number of new arguments");
                    return std::make_shared<Pattern>(this->get_predicate());
                }

                virtual std::string description() const override
                {
                    return "Pattern";
                } //@TODO [nikolayk] edit description to print out if the pattern is binded and if so the binded node

                virtual void propagate_types() override {}
                std::function<bool(std::shared_ptr<Node>)> get_predicate() const
                {
                    return m_predicate;
                }

            protected:
                std::function<bool(std::shared_ptr<Node>)> m_predicate;
            };
        }
    }
}
