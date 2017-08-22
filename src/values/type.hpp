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

#include <memory>
#include <vector>

#include "element_type.hpp"

namespace ngraph {

    class Shape
    {
    public:
        Shape(const std::initializer_list<size_t>& sizes)
        : m_sizes(sizes)
        {}

    protected:
        std::vector<size_t> m_sizes;
    };

    // ValueType is
    //   TensorViewType
    //   | TupleType(ValueType[])
    class ValueType
    {
    };

    class TensorViewType : public ValueType
    {
    public:
        TensorViewType(const ElementType& element_type, const Shape& shape)
        : m_element_type(element_type)
        , m_shape(shape)
        {}

    protected:
        TensorViewType(const TensorViewType&) = delete;
        const ElementType& m_element_type;
        Shape m_shape;
    };

    class TupleType : public ValueType
    {
    public:

        TupleType(const std::vector<std::shared_ptr<ValueType>>& element_types)
        : m_element_types(element_types)
        {}

    protected:
        std::vector<std::shared_ptr<ValueType>> m_element_types;
    };
}