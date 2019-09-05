//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF AnyOf KIND, either express or implied.
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
            /// \brief AnyOfs are used in patterns to express arbitrary queries on a node
            ///
            /// When AnyOf predicate matches a node; Matcher tries to match node's arguments to
            /// a single argument of AnyOf one by one. The first match is returned.
            /// This is useful for nodes with variable number of arguments such as Concat
            /// AnyOf enables on to specify one single branch/chain. The remaining arguments
            /// can be discovered (in a callback) by simply inspecting matched node's argument.
            class AnyOf : public Pattern
            {
            public:
                /// \brief creates a AnyOf node containing a sub-pattern described by \sa type and
                ///        \sa shape.
                AnyOf(const element::Type& type,
                      const PartialShape& s,
                      Predicate pred,
                      const NodeVector& wrapped_nodes)
                    : Pattern(wrapped_nodes, pred)
                {
                    if (!pred)
                    {
                        throw ngraph_error("predicate is required");
                    }

                    if (wrapped_nodes.size() != 1)
                    {
                        throw ngraph_error("AnyOf expects exactly one argument");
                    }
                    set_output_type(0, type, s);
                }

                /// \brief creates a AnyOf node containing a sub-pattern described by the type and
                ///        shape of \sa node.
                AnyOf(std::shared_ptr<Node> node, Predicate pred, const NodeVector& wrapped_nodes)
                    : AnyOf(node->get_element_type(),
                            node->get_output_partial_shape(0),
                            pred,
                            wrapped_nodes)
                {
                }

                const std::string& description() const override
                {
                    static std::string desc = "AnyOf";
                    return desc;
                }
            };
        }
    }
}
