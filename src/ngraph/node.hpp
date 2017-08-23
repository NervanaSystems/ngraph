// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#pragma once

#include <vector>

#include "ngraph/type.hpp"

namespace ngraph
{
    class Op;

    class Node : public TypedValueMixin
    {
    public:
        using ptr = std::shared_ptr<Node>;

        Node(const std::vector<Node::ptr>& arguments, ValueType::ptr type = 0)
            : m_arguments(arguments)
            , TypedValueMixin(type)
        {
        }

        virtual ~Node() {}
        virtual std::vector<Node::ptr> dependents() { return m_arguments; }

    protected:
        std::vector<Node::ptr> m_arguments;
    };

    class Call : public Node
    {
    public:
        virtual Op& op() const = 0;

    protected:
        Call(const std::vector<Node::ptr>& arguments)
            : Node(arguments, 0)
        {
        }
    };
}