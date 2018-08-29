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

#include <functional>

#include "ngraph/node.hpp"
#include "ngraph/pass/graph_rewrite.hpp"

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
                /// \brief \p a base class for \sa Skip and \sa Label
                ///
                Pattern(const std::string& type_name, const NodeVector& nodes, Predicate pred)
                    : Node(type_name, nodes)
                    , m_predicate(pred)
                {
                }

                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override
                {
                    throw ngraph_error("Uncopyable");
                }

                Predicate get_predicate() const { return m_predicate; }
            protected:
                std::function<bool(std::shared_ptr<Node>)> m_predicate;
            };
        }
    }
}
