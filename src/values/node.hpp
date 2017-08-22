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

#include "values/type.hpp"

namespace ngraph
{
    class Node
    {
    public:
        Node(const std::vector<std::shared_ptr<Node>>& arguments,
             std::shared_ptr<ValueType>                type = 0)
            : m_arguments(arguments)
            , m_type(type)
        {
        }

        virtual ~Node() {}
        virtual std::vector<std::shared_ptr<Node>> dependents() { return m_arguments; }

        void type(const std::shared_ptr<ValueType>& t) { m_type = t; }

        void type(const ElementType& element_type, const Shape& shape)
        {
            m_type = std::make_shared<TensorViewType>(element_type, shape);
        }

        std::shared_ptr<ValueType> type() const { return m_type; }

    protected:
        std::vector<std::shared_ptr<Node>> m_arguments;
        std::shared_ptr<ValueType>         m_type;
    };

    class Call : public Node
    {
    protected:
        Call(const std::vector<std::shared_ptr<Node>>& arguments)
            : Node(arguments, 0)
        {
        }
    };
}