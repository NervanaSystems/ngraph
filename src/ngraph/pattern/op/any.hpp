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
            /// \brief Anys are used in patterns to express arbitrary queries on a node
            class Any : public Pattern
            {
            public:
                /// \brief creates a Any node containing a sub-pattern described by \sa type and \sa
                ///        shape.
                Any(const element::Type& type,
                    const PartialShape& s,
                    Predicate pred,
                    const NodeVector& wrapped_nodes)
                    : Pattern(wrapped_nodes, pred)
                {
                    if (!pred)
                    {
                        throw ngraph_error("predicate is required");
                    }
                    set_output_type(0, type, s);
                }

                /// \brief creates a Any node containing a sub-pattern described by the type and
                ///        shape of \sa node.
                Any(std::shared_ptr<Node> node, Predicate pred, const NodeVector& wrapped_nodes)
                    : Any(node->get_element_type(),
                          node->get_output_partial_shape(0),
                          pred,
                          wrapped_nodes)
                {
                }

                const std::string& description() const override
                {
                    static std::string desc = "Any";
                    return desc;
                }
            };
        }
    }
}
