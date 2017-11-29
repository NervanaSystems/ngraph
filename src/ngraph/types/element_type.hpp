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

//================================================================================================
// ElementType
//================================================================================================

#pragma once

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>

#include "ngraph/common.hpp"
#include "ngraph/except.hpp"

namespace ngraph
{
    namespace element
    {
        class Type;

        extern const Type boolean;
        extern const Type f32;
        extern const Type f64;
        extern const Type i8;
        extern const Type i16;
        extern const Type i32;
        extern const Type i64;
        extern const Type u8;
        extern const Type u16;
        extern const Type u32;
        extern const Type u64;

        class Type
        {
        public:
            Type() = delete;
            Type(const Type&) = default;
            Type(size_t bitwidth, bool is_real, bool is_signed, const std::string& cname);
            virtual ~Type() {}
            const std::string& c_type_string() const;
            size_t size() const;
            size_t hash() const;
            bool is_real() const { return m_is_real; }
            bool is_signed() const { return m_is_signed; }
            size_t bitwidth() const { return m_bitwidth; }
            bool operator==(const Type& other) const;
            bool operator!=(const Type& other) const { return !(*this == other); }
            bool operator<(const Type& other) const;
            friend std::ostream& operator<<(std::ostream&, const Type&);

        private:
            static std::map<std::string, Type> m_element_list;
            size_t m_bitwidth;
            bool m_is_real;
            bool m_is_signed;
            const std::string m_cname;
        };

        template <typename T>
        const Type& from()
        {
            if (typeid(T) == typeid(char) || typeid(T) == typeid(bool))
            {
                return boolean;
            }
            else if (typeid(T) == typeid(float))
            {
                return f32;
            }
            else if (typeid(T) == typeid(double))
            {
                return f64;
            }
            else if (typeid(T) == typeid(int8_t))
            {
                return i8;
            }
            else if (typeid(T) == typeid(int16_t))
            {
                return i16;
            }
            else if (typeid(T) == typeid(int32_t))
            {
                return i32;
            }
            else if (typeid(T) == typeid(int64_t))
            {
                return i64;
            }
            else if (typeid(T) == typeid(uint8_t))
            {
                return u8;
            }
            else if (typeid(T) == typeid(uint16_t))
            {
                return u16;
            }
            else if (typeid(T) == typeid(uint32_t))
            {
                return u32;
            }
            else if (typeid(T) == typeid(uint64_t))
            {
                return u64;
            }
            else
            {
                throw std::invalid_argument("Unknown type");
            }
        }

        std::ostream& operator<<(std::ostream& out, const ngraph::element::Type& obj);

        // Provides a compile-time name for a C++ type.
        // Used in TraitedType for the string that supplies the C++ type name during code generation,
        // so it needs to be a valid C++ name.
        template <typename T>
        const char* traited_type_name()
        {
            throw ngraph_error("Unknown type");
        }

// Define a type string for a type T. Will make traited_type_name<T>() return "T"
#define NGRAPH_DEFINE_TRAITED_TYPE_NAME(T)                                                         \
    template <>                                                                                    \
    constexpr const char* traited_type_name<T>()                                                   \
    {                                                                                              \
        return #T;                                                                                 \
    }

        // Literals (and probably other things we don't know about yet) need to have their C++ types
        // and element types coordinated. Every element type corresponds to a TraitedType which provides
        // access to both the instance and the C++ type used to hold the value during compilation.
        template <typename T>
        class TraitedType : public Type
        {
            TraitedType(const TraitedType&) = delete;
            TraitedType& operator=(const TraitedType&) = delete;

        protected:
            TraitedType()
                : Type(sizeof(T) * 8,
                       std::is_floating_point<T>::value,
                       std::is_signed<T>::value,
                       traited_type_name<T>())
            {
            }

        public:
            // This is the C++ type used to hold a value of this element type during compilation
            using type = T;
            // This returns a reference to an instance of this element type.
            static const TraitedType<T>& element_type()
            {
                static TraitedType<T> t;
                return t;
            }
        };

        NGRAPH_DEFINE_TRAITED_TYPE_NAME(char)
        using Bool = TraitedType<char>;

        NGRAPH_DEFINE_TRAITED_TYPE_NAME(float)
        using Float32 = TraitedType<float>;

        NGRAPH_DEFINE_TRAITED_TYPE_NAME(double)
        using Float64 = TraitedType<double>;

        NGRAPH_DEFINE_TRAITED_TYPE_NAME(int8_t)
        using Int8 = TraitedType<int8_t>;

        NGRAPH_DEFINE_TRAITED_TYPE_NAME(int32_t)
        using Int32 = TraitedType<int32_t>;

        NGRAPH_DEFINE_TRAITED_TYPE_NAME(int64_t)
        using Int64 = TraitedType<int64_t>;

        NGRAPH_DEFINE_TRAITED_TYPE_NAME(uint8_t)
        using UInt8 = TraitedType<uint8_t>;

        NGRAPH_DEFINE_TRAITED_TYPE_NAME(uint32_t)
        using UInt32 = TraitedType<uint32_t>;

        NGRAPH_DEFINE_TRAITED_TYPE_NAME(uint64_t)
        using UInt64 = TraitedType<uint64_t>;
    }
}
