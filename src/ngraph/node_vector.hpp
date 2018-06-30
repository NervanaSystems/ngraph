/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <memory>
#include <vector>

namespace ngraph
{
    class Node;

    namespace op
    {
        class Result;
    }

    /// \brief Zero or more nodes.
    class NodeVector : public std::vector<std::shared_ptr<Node>>
    {
    public:
        NodeVector(const std::initializer_list<std::shared_ptr<Node>>& nodes)
            : std::vector<std::shared_ptr<Node>>(nodes)
        {
        }

        NodeVector(const std::vector<std::shared_ptr<Node>>& nodes)
            : std::vector<std::shared_ptr<Node>>(nodes)
        {
        }

        NodeVector(const NodeVector& nodes)
            : std::vector<std::shared_ptr<Node>>(nodes)
        {
        }

        NodeVector(size_t size)
            : std::vector<std::shared_ptr<Node>>(size)
        {
        }

        NodeVector& operator=(const NodeVector& other) = default;

        NodeVector() {}
    };
}
