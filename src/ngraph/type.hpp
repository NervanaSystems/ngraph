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
#include "shape.hpp"

namespace ngraph
{
    class TensorViewType;
    class TupleType;

    /**
     ** ValueType is
     **   TensorViewType
     **   | TupleType(ValueType[])
     **/
    class ValueType
    {
    public:
        /**
         ** Preferred handle
         **/
        using ptr = std::shared_ptr<ValueType>;

        virtual ~ValueType() {}
        virtual bool operator==(const ValueType::ptr& that) const = 0;
        bool         operator!=(const ValueType::ptr& that) const { return !(*this == that); }
    };

    /**
     ** Describes a tensor view; an element type and a shape.
     **/
    class TensorViewType : public ValueType
    {
    public:
        /**
         ** Preferred handle
         **/
        using ptr = std::shared_ptr<TensorViewType>;

        /**
         ** /param element_type The type of the tensor elements.
         ** /param shape The shape of the tensor.
         **/
        TensorViewType(const element::Type& element_type, const Shape& shape)
            : m_element_type(element_type)
            , m_shape(shape)
        {
        }

        const element::Type& get_element_type() const { return m_element_type; }
        const Shape&         get_shape() const { return m_shape; }

        virtual bool operator==(const ValueType::ptr& that) const override;

    protected:
        const element::Type& m_element_type;
        Shape                m_shape;
    };

    /**
     ** Describes a tuple of values; a vector of types
     **/
    class TupleType : public ValueType
    {
    public:
        /**
         ** The preferred handle
         **/
        using ptr = std::shared_ptr<ValueType>;

        /**
         ** Construct empty tuple and add value types later.
         **/
        TupleType() {}
        /**
         ** /param element_types A vector of types for the tuple elements
         **/
        TupleType(const std::vector<ValueType::ptr>& element_types)
            : m_element_types(element_types)
        {
        }

        const std::vector<ValueType::ptr> get_element_types() const { return m_element_types; }
        std::vector<ValueType::ptr>       set_element_types() { return m_element_types; }

        virtual bool operator==(const ValueType::ptr& that) const override;

    protected:
        std::vector<ValueType::ptr> m_element_types;
    };

    /**
     ** Mixin for objects with type information
     **/
    class TypedValueMixin
    {
    public:
        TypedValueMixin(const ValueType::ptr& value_type = nullptr)
            : m_value_type(value_type)
        {
        }

        /**
         ** Set the type
         ** /param type The new type
         **/
        void set_value_type(const ValueType::ptr& value_type) { m_value_type = value_type; }
        /**
         ** Set the type to be a tensor view type
         ** /param element_type The type of the tensor elements
         ** /param shape The shape of the view
         **/
        void set_value_type(const element::Type& element_type, const Shape& shape)
        {
            m_value_type = std::make_shared<TensorViewType>(element_type, shape);
        }

        /**
         ** The type associated with this value.
         **/
        ValueType::ptr get_value_type() { return m_value_type; }
        /**
         ** The type associated with this value.
         **/
        const ValueType::ptr get_value_type() const { return m_value_type; }
    protected:
        ValueType::ptr m_value_type;
    };
}
