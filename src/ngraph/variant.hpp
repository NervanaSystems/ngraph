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

#include <string>

#include "ngraph/ngraph_visibility.hpp"
#include "ngraph/type.hpp"

namespace ngraph
{
    using VariantTypeInfo = DiscreteTypeInfo;

    class Variant
    {
    public:
        virtual ~Variant() {}
        virtual const VariantTypeInfo& get_type_info() const = 0;
    };

    template <typename VT>
    class VariantImpl : public Variant
    {
    public:
        using value_type = VT;

        VariantImpl(const value_type& value)
            : m_value(value)
        {
        }
        const value_type& get() const { return m_value; }
        value_type& get() { return m_value; }
        void set(const value_type& value) { m_value = value; }
    protected:
        value_type m_value;
    };

    template <typename VT>
    class VariantWrapper
    {
    };

    template <>
    class VariantWrapper<std::string> : public VariantImpl<std::string>
    {
    public:
        NGRAPH_API
        static constexpr VariantTypeInfo type_info{"Variant::std::string", 0};
        const VariantTypeInfo& get_type_info() const override { return type_info; }
        VariantWrapper(const value_type& value)
            : VariantImpl<value_type>(value)
        {
        }
    };

    template <>
    class VariantWrapper<int64_t> : public VariantImpl<int64_t>
    {
    public:
        NGRAPH_API
        static constexpr VariantTypeInfo type_info{"Variant::int64_t", 0};
        const VariantTypeInfo& get_type_info() const override { return type_info; }
        VariantWrapper(const value_type& value)
            : VariantImpl<value_type>(value)
        {
        }
    };
}
