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
            /// \brief \p Skip allows users to specify unexpected nodes in a pattern
            /// and skip them if a predicate condition is satisfied.
            ///
            class Skip : public Pattern
            {
            public:
                Skip(const std::shared_ptr<Node>& arg, Predicate predicate = nullptr)
                    : Pattern("Skip", NodeVector{arg}, predicate)
                {
                    set_output_type(0, arg->get_element_type(), arg->get_shape());
                }
            };
        }
    }
}
