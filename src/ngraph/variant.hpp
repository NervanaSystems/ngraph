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

#define VARIANT_NAME(VT, VERSION) ngraph::VariantImpl<VT, VERSION>

// Convenience macro to declare a variant for values of type VT
#define DECLARE_VARIANT(VT, API, NAME, VERSION)                                                    \
    template <>                                                                                    \
    class VARIANT_NAME(VT, VERSION)                                                                \
        : public ngraph::Variant                                                                   \
    {                                                                                              \
    public:                                                                                        \
        VariantImpl() = default;                                                                   \
        VariantImpl(const VT& value)                                                               \
            : m_value(value)                                                                       \
        {                                                                                          \
        }                                                                                          \
        API static constexpr ngraph::VariantTypeInfo type_info{NAME, VERSION};                     \
        const ngraph::VariantTypeInfo& get_type_info() const override { return type_info; }        \
        const VT& get() const { return m_value; }                                                  \
        VT& get() { return m_value; }                                                              \
        void set(const VT& value) { m_value = value; }                                             \
    private:                                                                                       \
        VT m_value;                                                                                \
    };

// Convenience macro to define a variant for values of type VT
#define DEFINE_VARIANT(VT, VERSION)                                                                \
    constexpr ngraph::VariantTypeInfo VARIANT_NAME(VT, VERSION)::type_info;

namespace ngraph
{
    using VariantTypeInfo = DiscreteTypeInfo;

    class Variant
    {
    public:
        virtual ~Variant() {}
        virtual const VariantTypeInfo& get_type_info() const = 0;
    };

    // Hand-constructed variant
    class StringVariant : public Variant
    {
    public:
        NGRAPH_API
        static constexpr ngraph::VariantTypeInfo type_info{"Variant::StringVariant", 0};
        const VariantTypeInfo& get_type_info() const override { return type_info; }
        StringVariant(const std::string& value)
            : m_value(value)
        {
        }
        StringVariant() = default;
        const std::string& get() const { return m_value; }
        std::string& get() { return m_value; }
        void set(const std::string& value) { m_value = value; }
    private:
        std::string m_value;
    };

    template <typename VT, int version>
    class VariantImpl : public Variant
    {
    };

    // Declare a variant for std::string
    DECLARE_VARIANT(std::string, NGRAPH_API, "Variant::std::string", 0)
    // Declare a variant for uint64_t
    DECLARE_VARIANT(uint64_t, NGRAPH_API, "Variant::uint64_t", 0)
}
