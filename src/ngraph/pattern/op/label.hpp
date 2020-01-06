//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include "ngraph/node.hpp"
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
            class NGRAPH_API Label : public Pattern
            {
            public:
                static constexpr NodeTypeInfo type_info{"patternLabel", 0};
                const NodeTypeInfo& get_type_info() const override;
                /// \brief creates a Label node containing a sub-pattern described by \sa type and
                ///        \sa shape.
                ///
                /// this Label node can be bound only to the nodes in the input graph
                /// that match the pattern specified by \sa wrapped_nodes
                /// Example:
                /// \code{.cpp}
                /// auto add = a + b; // a and b are op::Parameter in this example
                /// auto label = std::make_shared<pattern::op::Label>(element::f32,
                ///                                                   Shape{2,2},
                ///                                                   nullptr,
                ///                                                   OutputVector{add});
                /// \endcode
                Label(const element::Type& type,
                      const PartialShape& s,
                      Predicate pred,
                      const OutputVector& wrapped_values)
                    : Pattern(wrapped_values, pred)
                {
                    set_output_type(0, type, s);
                }

                Label(const element::Type& type, const PartialShape& s)
                    : Label(type, s, [](std::shared_ptr<Node>) { return true; }, OutputVector())
                {
                }

                Label(const element::Type& type, const PartialShape& s, Predicate pred)
                    : Label(type, s, pred, OutputVector{})
                {
                }

                Label(const element::Type& type,
                      const PartialShape& s,
                      Predicate pred,
                      const NodeVector& wrapped_values)
                    : Label(type, s, pred, as_output_vector(wrapped_values))
                {
                }

                /// \brief creates a Label node containing a sub-pattern described by the type and
                ///        shape of \sa node.
                ///
                /// this Label node can be bound only to the nodes in the input graph
                /// that match the pattern specified by \sa wrapped_values
                /// Example:
                /// \code{.cpp}
                /// auto add = a + b; // a and b are op::Parameter in this example
                /// auto label = std::make_shared<pattern::op::Label>(add,
                ///                                                   nullptr,
                ///                                                   OutputVector{add});
                /// \endcode
                Label(std::shared_ptr<Node> node,
                      Predicate pred,
                      const OutputVector& wrapped_values)
                    : Label(node->get_element_type(),
                            node->get_output_partial_shape(0),
                            pred,
                            wrapped_values)
                {
                }
                Label(std::shared_ptr<Node> node, Predicate pred)
                    : Label(node->get_element_type(),
                            node->get_output_partial_shape(0),
                            pred,
                            OutputVector{})
                {
                }

                Label(std::shared_ptr<Node> node)
                    : Label(node->get_element_type(),
                            node->get_output_partial_shape(0),
                            [](std::shared_ptr<Node>) { return true; },
                            OutputVector{})
                {
                }
                Label(std::shared_ptr<Node> node, Predicate pred, const NodeVector& wrapped_values)
                    : Label(node->get_element_type(),
                            node->get_output_partial_shape(0),
                            pred,
                            as_output_vector(wrapped_values))
                {
                }
            };
        }
    }
}
