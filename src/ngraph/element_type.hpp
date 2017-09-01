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
#include <string>
#include <type_traits>

#include "except.hpp"

namespace ngraph
{
    namespace element
    {
        class Type
        {
        public:
            Type(size_t bitwidth, bool is_float, bool is_signed, const std::string& cname);

            const std::string& c_type_string() const;
            size_t             size() const;
            size_t             hash() const
            {
                std::hash<std::string> h;
                return h(m_cname);
            }

            bool operator==(const Type& other) const;
            bool operator!=(const Type& other) const { return !(*this == other); }
        private:
            static std::map<std::string, Type> m_element_list;
            size_t            m_bitwidth;
            bool              m_is_float;
            bool              m_is_signed;
            const std::string m_cname;
        };

        // Provides a compile-time name for a C++ type.
        // Used in TraitedType for the string that supplies the C++ type name.
        template<typename T>
        const char* traited_type_name()
        {
            throw ngraph_error("Unkmown type");
        }

        // Literals (and probably other things we don't know about yet) need to have their C++ types
        // and element types coordinated. Every element type corresponds to a TraitedType which provides
        // access to both the instance and the C++ type used to hold the value during compilation.
        template <typename T>
        class TraitedType : public Type
        {
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
            static const TraitedType<T>& element_type(){
                static TraitedType<T> t;
                return t;
            }
        };

        template<>
        constexpr const char* traited_type_name<float>()
        {
            return "float";
        }

        using Float = TraitedType<float>;


        template<>
        constexpr const char* traited_type_name<int8_t>()
        {
            return "int8_t";
        }

        using Int8 = TraitedType<int8_t>;

        template<>
        constexpr const char* traited_type_name<int32_t>()
        {
            return "int32_t";
        }

        using Int32 = TraitedType<int32_t>;

        template<>
        constexpr const char* traited_type_name<int64_t>()
        {
            return "int64_t";
        }

        using Int64 = TraitedType<int64_t>;

        template<>
        constexpr const char* traited_type_name<uint8_t>()
        {
            return "uint8_t";
        }

        using UInt8 = TraitedType<uint8_t>;

        template<>
        constexpr const char* traited_type_name<uint32_t>()
        {
            return "uint32_t";
        }

        using UInt32 = TraitedType<uint32_t>;

        template<>
        constexpr const char* traited_type_name<uint64_t>()
        {
            return "uint64_t";
        }
        
        using UInt64 = TraitedType<uint64_t>;
    }
}
