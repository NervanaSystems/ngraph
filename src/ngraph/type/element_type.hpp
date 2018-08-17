/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

//================================================================================================
// ElementType
//================================================================================================

#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "ngraph/except.hpp"

namespace ngraph
{
    namespace element
    {
        class Type;

        extern const Type unspecified;
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
            Type() {}
            Type(const Type&) = default;
            Type(size_t bitwidth, bool is_real, bool is_signed, const std::string& cname);
            Type& operator=(const Type&);
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
            static std::vector<const Type*> get_known_types();

            /// Returns true if the type is floating point, else false.
            bool get_is_real() const { return m_is_real; }
        private:
            size_t m_bitwidth{0};
            bool m_is_real{false};
            bool m_is_signed{false};
            std::string m_cname{"unspecified"};
        };

        template <typename T>
        const Type& from()
        {
            throw std::invalid_argument("Unknown type");
        }
        template <>
        const Type& from<char>();
        template <>
        const Type& from<bool>();
        template <>
        const Type& from<float>();
        template <>
        const Type& from<double>();
        template <>
        const Type& from<int8_t>();
        template <>
        const Type& from<int16_t>();
        template <>
        const Type& from<int32_t>();
        template <>
        const Type& from<int64_t>();
        template <>
        const Type& from<uint8_t>();
        template <>
        const Type& from<uint16_t>();
        template <>
        const Type& from<uint32_t>();
        template <>
        const Type& from<uint64_t>();

        std::ostream& operator<<(std::ostream& out, const ngraph::element::Type& obj);
    }
}
