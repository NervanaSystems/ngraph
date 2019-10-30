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

#include "ngraph/enum_names.hpp"
#include "ngraph/type.hpp"

namespace ngraph
{
    /// Provides a generic way to access attribute values for serialization
    class AttributeAdapter
    {
    public:
        virtual ~AttributeAdapter() {}
        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter", 0};
        virtual const DiscreteTypeInfo& get_type_info() const { return type_info; }
        /// Returns the value as a string
        virtual std::string get_string() const = 0;
        /// 
        virtual void set_string(const std::string& value) const = 0;
    };

    template <typename Type>
    class TypeAdapter : public AttributeAdapter
    {
    public:
        operator Type&() const { return m_value; }
    protected:
        TypeAdapter(Type& value)
            : m_value(value)
        {
        }
        Type& m_value;
    };

    template <typename Type>
    class EnumAdapter : public TypeAdapter<Type>
    {
    public:
        EnumAdapter(Type& value)
            : TypeAdapter<Type>(value)
        {
        }
        static const DiscreteTypeInfo type_info;
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        std::string get_string() const override { return as_string(TypeAdapter<Type>::m_value); }
        void set_string(const std::string& value) const override
        {
            TypeAdapter<Type>::m_value = as_enum<Type>(value);
        }
    };

    template <typename Type>
    class ObjectAdapter : public TypeAdapter<Type>
    {
    public:
        ObjectAdapter(Type& value)
            : TypeAdapter<Type>(value)
        {
        }
        static const DiscreteTypeInfo type_info;
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        std::string get_string() const override { return "TODO"; }
        void set_string(const std::string& value) const override {}
    };
}
