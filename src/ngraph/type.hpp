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

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ngraph/ngraph_visibility.hpp"

namespace ngraph
{
    /// Supports three functions, is_type<Type>, as_type<Type>, and as_type_ptr<Type> for type-safe
    /// dynamic conversions via static_cast/static_ptr_cast without using C++ RTTI.
    /// Type must have a static constexpr type_info member and a virtual get_type_info() member that
    /// returns a reference to its type_info member.

    /// Type information for a type system without inheritance; instances have exactly one type not
    /// related to any other type.
    struct NGRAPH_API DiscreteTypeInfo
    {
        const char* name;
        uint64_t version;

        bool is_castable(const DiscreteTypeInfo& target_type) const { return *this == target_type; }
        // For use as a key
        bool operator<(const DiscreteTypeInfo& b) const
        {
            return version < b.version || (version == b.version && strcmp(name, b.name) < 0);
        }
        bool operator<=(const DiscreteTypeInfo& b) const
        {
            return version < b.version || (version == b.version && strcmp(name, b.name) <= 0);
        }
        bool operator>(const DiscreteTypeInfo& b) const
        {
            return version < b.version || (version == b.version && strcmp(name, b.name) > 0);
        }
        bool operator>=(const DiscreteTypeInfo& b) const
        {
            return version < b.version || (version == b.version && strcmp(name, b.name) >= 0);
        }
        bool operator==(const DiscreteTypeInfo& b) const
        {
            return version == b.version && strcmp(name, b.name) == 0;
        }
        bool operator!=(const DiscreteTypeInfo& b) const
        {
            return version != b.version || strcmp(name, b.name) != 0;
        }
    };

    /// \brief Tests if value is a pointer/shared_ptr that can be statically cast to a
    /// Type*/shared_ptr<Type>
    template <typename Type, typename Value>
    typename std::enable_if<
        std::is_convertible<
            decltype(std::declval<Value>()->get_type_info().is_castable(Type::type_info)),
            bool>::value,
        bool>::type
        is_type(Value value)
    {
        return value->get_type_info().is_castable(Type::type_info);
    }

    /// Casts a Value* to a Type* if it is of type Type, nullptr otherwise
    template <typename Type, typename Value>
    typename std::enable_if<
        std::is_convertible<decltype(static_cast<Type*>(std::declval<Value>())), Type*>::value,
        Type*>::type
        as_type(Value value)
    {
        return is_type<Type>(value) ? static_cast<Type*>(value) : nullptr;
    }

    /// Casts a std::shared_ptr<Value> to a std::shared_ptr<Type> if it is of type
    /// Type, nullptr otherwise
    template <typename Type, typename Value>
    typename std::enable_if<
        std::is_convertible<decltype(std::static_pointer_cast<Type>(std::declval<Value>())),
                            std::shared_ptr<Type>>::value,
        std::shared_ptr<Type>>::type
        as_type_ptr(Value value)
    {
        return is_type<Type>(value) ? std::static_pointer_cast<Type>(value)
                                    : std::shared_ptr<Type>();
    }
}

namespace std
{
    template <>
    struct hash<ngraph::DiscreteTypeInfo>
    {
        size_t operator()(const ngraph::DiscreteTypeInfo& k) const;
    };
}
