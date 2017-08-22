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

#include "values/node.hpp"
#include "values/op.hpp"
#include "values/type.hpp"

namespace ngraph
{
    class Function;

    class Parameter : public Node
    {
    public:
        Parameter(Function& function, size_t index, const std::shared_ptr<ValueType>& type)
            : Node({}, type)
            , m_function(function)
            , m_index(index)
        {
        }

    protected:
        Function& m_function;
        size_t    m_index;
    };

    class Result
    {
    public:
        void type(const std::shared_ptr<ValueType>& t) { m_type = t; }

        void type(const ElementType& element_type, const Shape& shape)
        {
            m_type = std::make_shared<TensorViewType>(element_type, shape);
        }

        std::shared_ptr<ValueType> type() const { return m_type; }

        std::shared_ptr<Node> value() const { return m_value; }
        void                  value(const std::shared_ptr<Node>& value) { m_value = value; }

    protected:
        std::shared_ptr<ValueType> m_type;
        std::shared_ptr<Node>      m_value;
    };

    class Function
    {
    public:
        Function(size_t n_parameters)
            : m_parameters(n_parameters)
        {
        }

        Result* result() { return &m_result; }

        std::shared_ptr<Parameter> parameter(size_t i) { return m_parameters[i]; }

    protected:
        std::vector<std::shared_ptr<Parameter>> m_parameters;
        Result                                  m_result;
    };

} // end namespace ngraph