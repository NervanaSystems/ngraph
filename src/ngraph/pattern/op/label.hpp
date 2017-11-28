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
            /// \brief Labels are used in patterns to express repeating nodes in an input graph
            /// and bind them to specific nodes from the graph
            ///
            class Label : public Pattern
            {
            public:
                static std::shared_ptr<Label>
                    make_from_node(const std::shared_ptr<ngraph::Node>& node,
                                   Predicate pred = nullptr)
                {
                    auto label = std::make_shared<Label>(pred);
                    label->set_value_type_checked(node->get_value_type());
                    return label;
                }
                bool is_bound() { return m_bound != nullptr; }
                std::shared_ptr<Node> get_bound_node() { return m_bound; }
                void reset() { m_bound.reset(); }
                void bind(std::shared_ptr<Node> n) { m_bound = n; }
                Label(Predicate pred = nullptr)
                    : Pattern("Label", Nodes{}, pred)
                {
                }

            private:
                std::shared_ptr<Node> m_bound;
            };
        }
    }
}
