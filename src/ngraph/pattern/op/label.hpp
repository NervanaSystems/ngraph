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
            /// Fails if the predicate returns false on the graph value.
            ///
            /// The graph value is added to the matched values list. If the Label is already
            /// associated with a value, the match succeeds if the value is the same as the graph
            /// value. Otherwise, the label is associated with the graph value and the match
            /// succeeds if the pattern input matches the graph value.
            ///
            /// DEPRECATED: If no inputs are given to Label, a True node is serves as the input. If
            /// more than one inputs are given, an Or pattern of the inputs serves as the input.
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
                      const ValuePredicate pred,
                      const OutputVector& wrapped_values)
                    : Pattern(OutputVector{wrap_values(wrapped_values)}, pred)
                {
                    set_output_type(0, type, s);
                }

                Label(const element::Type& type, const PartialShape& s)
                    : Label(type, s, [](const Output<Node>&) { return true; }, OutputVector())
                {
                }

                Label(const element::Type& type, const PartialShape& s, ValuePredicate pred)
                    : Label(type, s, pred, OutputVector{})
                {
                }

                Label(const element::Type& type, const PartialShape& s, NodePredicate pred)
                    : Label(type, s, as_value_predicate(pred), OutputVector{})
                {
                }

                Label(const element::Type& type,
                      const PartialShape& s,
                      const NodePredicate pred,
                      const NodeVector& wrapped_values)
                    : Label(type, s, as_value_predicate(pred), as_output_vector(wrapped_values))
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
                Label(const Output<Node>& value,
                      const ValuePredicate pred,
                      const OutputVector& wrapped_values)
                    : Label(
                          value.get_element_type(), value.get_partial_shape(), pred, wrapped_values)
                {
                }
                Label(const Output<Node>& value, const ValuePredicate pred)
                    : Label(
                          value.get_element_type(), value.get_partial_shape(), pred, OutputVector{})
                {
                }

                Label(const Output<Node>& value, const NodePredicate pred)
                    : Label(value.get_element_type(),
                            value.get_partial_shape(),
                            as_value_predicate(pred),
                            OutputVector{})
                {
                }
                Label(const Output<Node>& value)
                    : Label(value.get_element_type(),
                            value.get_partial_shape(),
                            [](const Output<Node>&) { return true; },
                            OutputVector{})
                {
                }
                Label(const Output<Node>& node,
                      const NodePredicate pred,
                      const NodeVector& wrapped_values)
                    : Label(node.get_element_type(),
                            node.get_partial_shape(),
                            as_value_predicate(pred),
                            as_output_vector(wrapped_values))
                {
                }

                bool match_value(Matcher* matcher,
                                 const Output<Node>& pattern_value,
                                 const Output<Node>& graph_value) override;

            protected:
                static Output<Node> wrap_values(const OutputVector& wrapped_values);
            };
        }
    }
}
