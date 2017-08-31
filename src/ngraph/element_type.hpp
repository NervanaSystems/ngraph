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
            size_t                             m_bitwidth;
            bool                               m_is_float;
            bool                               m_is_signed;
            const std::string                  m_cname;
        };

        // Literals (and probably other things we don't know about yet) need to have their C++ types
        // and element types coordinated. Every element type corresponds to a TraitedType which provides
        // access to both the instance and the C++ type used to hold the value during compilation.
        template <typename T>
        class TraitedType : public Type
        {
        public:
            // This is the C++ type used to hold a value of this element type during compilation
            using ctype = T;
            // This is a reference to an instance of this element type.
            static const TraitedType<T>& type;

            TraitedType(const std::string& cname)
                : Type(sizeof(T) * 8,
                       std::is_floating_point<T>::value,
                       std::is_signed<T>::value,
                       cname)
            {
            }
        };

        // Human-readable names for the element types
        using Float  = TraitedType<float>;
        using Int8   = TraitedType<int8_t>;
        using Int32  = TraitedType<int32_t>;
        using Int64  = TraitedType<int64_t>;
        using UInt8  = TraitedType<uint8_t>;
        using UInt32 = TraitedType<uint32_t>;
        using UInt64 = TraitedType<uint64_t>;
    }
}
