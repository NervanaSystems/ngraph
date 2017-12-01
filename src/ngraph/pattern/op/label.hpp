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
                /// \brief creates a Label node from \sa node.
                ///
                /// this Label node can be bound to arbitrary nodes in an input graph
                /// as long as provided \sa pred is satisfied and the node hasn't been previously bound to
                /// the different node in the input graph
                /// \code{.cpp}
                /// auto pattern = pattern::op::Label::make_from_node(a); //a is op::Parameter
                /// matcher.match(pattern, a));
                /// \endcode
                static std::shared_ptr<Label>
                    make_from_node(const std::shared_ptr<ngraph::Node>& node,
                                   Predicate pred = nullptr)
                {
                    auto label = std::make_shared<Label>(Nodes{}, pred);
                    label->set_value_type_checked(node->get_value_type());
                    return label;
                }

                /// \brief creates a Label node containing a sub-pattern described by \sa node.
                ///
                /// this Label node can be bound only to the nodes in the input graph
                /// that match the pattern specified by \sa node
                /// Example:
                /// \code{.cpp}
                /// auto add = a + b; //a and b are op::Parameter in this example
                /// auto label = pattern::op::Label::wrap(add);
                /// \endcode
                static std::shared_ptr<Label> wrap(const std::shared_ptr<ngraph::Node>& node,
                                                   Predicate pred = nullptr)
                {
                    auto label = std::make_shared<Label>(Nodes{node}, pred);
                    label->set_value_type_checked(node->get_value_type());
                    return label;
                }

                Label(const Nodes& subgraph, Predicate pred)
                    : Pattern("Label", Nodes{subgraph}, pred)
                {
                }
            };
        }
    }
}
