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
            /// \brief Branches are used to allow repeat patterns
            class NGRAPH_API Branch : public Pattern
            {
            public:
                static constexpr NodeTypeInfo type_info{"patternBranch", 0};
                const NodeTypeInfo& get_type_info() const override;
                /// \brief Creates a Branch pattern
                /// \param pattern the repeating pattern
                /// \param labels Labels where the repeat may occur
                Branch()
                    : Pattern(OutputVector{})
                {
                    set_output_type(0, element::f32, Shape{});
                }

                void set_repeat(const Output<Node>& repeat)
                {
                    m_repeat_node = repeat.get_node();
                    m_repeat_index = repeat.get_index();
                }

                Output<Node> get_repeat() const
                {
                    return m_repeat_node == nullptr
                               ? Output<Node>()
                               : Output<Node>{m_repeat_node->shared_from_this(), m_repeat_index};
                }

                bool match_value(pattern::Matcher* matcher,
                                 const Output<Node>& pattern_value,
                                 const Output<Node>& graph_value) override;

            protected:
                Node* m_repeat_node{nullptr};
                size_t m_repeat_index{0};
            };
        }
    }
}
