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

#include <memory>
#include <vector>

#include "ngraph/node_output.hpp"

namespace ngraph
{
    /// \brief Zero or more node outputs.
    class OutputVector : public std::vector<NodeOutput>
    {
    public:
        OutputVector(const std::initializer_list<NodeOutput>& outputs)
            : std::vector<NodeOutput>(outputs)
        {
        }

        OutputVector(const std::initializer_list<std::shared_ptr<Node>>& nodes);

        OutputVector(const std::vector<std::shared_ptr<Node>>& nodes);

        OutputVector(const OutputVector& outputs)
            : std::vector<NodeOutput>(outputs)
        {
        }

        OutputVector& operator=(const OutputVector& other) = default;

        OutputVector() {}
    };
}
