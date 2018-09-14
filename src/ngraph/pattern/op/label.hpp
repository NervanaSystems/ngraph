//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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
            class Label : public Pattern
            {
            public:
                /// \brief creates a Label node containing a sub-pattern described by \sa type and \sa shape.
                ///
                /// this Label node can be bound only to the nodes in the input graph
                /// that match the pattern specified by \sa wrapped_nodes
                /// Example:
                /// \code{.cpp}
                /// auto add = a + b; // a and b are op::Parameter in this example
                /// auto label = std::make_shared<pattern::op::Label>(element::f32, Shape{2,2} , nullptr, NodeVector{add});
                /// \endcode
                Label(const element::Type& type,
                      const Shape s,
                      Predicate pred = nullptr,
                      const NodeVector& wrapped_nodes = NodeVector{})
                    : Pattern("Label", wrapped_nodes, pred)
                {
                    set_output_type(0, type, s);
                }

                /// \brief creates a Label node containing a sub-pattern described by the type and shape of \sa node.
                ///
                /// this Label node can be bound only to the nodes in the input graph
                /// that match the pattern specified by \sa wrapped_nodes
                /// Example:
                /// \code{.cpp}
                /// auto add = a + b; // a and b are op::Parameter in this example
                /// auto label = std::make_shared<pattern::op::Label>(add, nullptr, NodeVector{add});
                /// \endcode
                Label(std::shared_ptr<Node> node,
                      Predicate pred = nullptr,
                      const NodeVector& wrapped_nodes = NodeVector{})
                    : Label(node->get_element_type(), node->get_shape(), pred, wrapped_nodes)
                {
                }
            };
        }
    }
}
