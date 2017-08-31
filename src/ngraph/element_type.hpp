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

        // Literals (and probably other things we don't know about yet) need to have their C++ types
        // and element types coordinated. Every element type corresponds to a TraitedType which provides
        // access to both the instance and the C++ type used to hold the value during compilation.
        template <typename T, typename U>
        class TraitedType : public Type
        {
        protected:
            TraitedType(const std::string& cname)
            : Type(sizeof(T) * 8,
                   std::is_floating_point<T>::value,
                   std::is_signed<T>::value,
                   cname)
            {
            }

        public:
            // This is the C++ type used to hold a value of this element type during compilation
            using ctype = T;
            // This is a reference to an instance of this element type.
            static const U& element_type(){
                static U t;
                return t;
            }
        };

        class Float : public TraitedType<float, Float>
        {
            friend class TraitedType<float, Float>;
            Float()
                : TraitedType<float, Float>("float")
                {
                }
        };

        class Int8 : public TraitedType<int8_t, Int8>
        {
            friend class TraitedType<int8_t, Int8>;
            Int8()
                : TraitedType<int8_t, Int8>("int8_t")
                {
                }
        };
        
        class Int32 : public TraitedType<int32_t, Int32>
        {
            friend class TraitedType<int32_t, Int32>;
            Int32()
                : TraitedType<int32_t, Int32>("int32_t")
                {
                }
        };

        class Int64 : public TraitedType<int64_t, Int64>
        {
            friend class TraitedType<int64_t, Int64>;
            Int64()
                : TraitedType<int64_t, Int64>("int64_t")
                {
                }
        };

        class UInt8 : public TraitedType<uint8_t, UInt8>
        {
            friend class TraitedType<uint8_t, UInt8>;
            UInt8()
                : TraitedType<uint8_t, UInt8>("uint8_t")
                {
                }
        };
        
        class UInt32 : public TraitedType<uint32_t, UInt32>
        {
            friend class TraitedType<uint32_t, UInt32>;
            UInt32()
                : TraitedType<uint32_t, UInt32>("uint32_t")
                {
                }
        };

        class UInt64 : public TraitedType<uint64_t, UInt64>
        {
            friend class TraitedType<uint64_t, UInt64>;
            UInt64()
                : TraitedType<uint64_t, UInt64>("uint64_t")
                {
                }
        };
    }
}
