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

//================================================================================================
// ElementType
//================================================================================================

#pragma once

#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "ngraph/except.hpp"
#include "ngraph/ngraph_visibility.hpp"
#include "ngraph/type/bfloat16.hpp"

namespace ngraph
{
    namespace element
    {
        enum class Type_t
        {
            undefined,
            dynamic,
            boolean,
            bf16,
            f32,
            f64,
            i8,
            i16,
            i32,
            i64,
            u8,
            u16,
            u32,
            u64
        };

        class Type
        {
        public:
            Type()
                : m_type{element::Type_t::undefined}
            {
            }
            Type(const Type&) = default;
            Type(const Type_t t)
                : m_type{t}
            {
            }
            Type(size_t bitwidth,
                 bool is_real,
                 bool is_signed,
                 bool is_quantized,
                 const std::string& cname);
            ~Type() {}
            Type& operator=(const Type&) = default;
            Type_t get_type_enum() const { return m_type; }
            const std::string& c_type_string() const;
            size_t size() const;
            size_t hash() const;
            bool is_static() const;
            bool is_dynamic() const { return !is_static(); }
            bool is_real() const;
            bool is_signed() const;
            bool is_quantized() const;
            size_t bitwidth() const;
            bool operator==(const Type& other) const;
            bool operator!=(const Type& other) const { return !(*this == other); }
            bool operator<(const Type& other) const;
            friend std::ostream& operator<<(std::ostream&, const Type&);
            static std::vector<const Type*> get_known_types();

            /// \brief Checks whether this element type is merge-compatible with `t`.
            /// \param t The element type to compare this element type to.
            /// \return `true` if this element type is compatible with `t`, else `false`.
            bool compatible(const element::Type& t) const;

            /// \brief Merges two element types t1 and t2, writing the result into dst and
            ///        returning true if successful, else returning false.
            ///
            ///        To "merge" two element types t1 and t2 is to find the least restrictive
            ///        element type t that is no more restrictive than t1 and t2, if t exists.
            ///        More simply:
            ///
            ///           merge(dst,element::Type::dynamic,t)
            ///              writes t to dst and returns true
            ///
            ///           merge(dst,t,element::Type::dynamic)
            ///              writes t to dst and returns true
            ///
            ///           merge(dst,t1,t2) where t1, t2 both static and equal
            ///              writes t1 to dst and returns true
            ///
            ///           merge(dst,t1,t2) where t1, t2 both static and unequal
            ///              does nothing to dst, and returns false
            static bool merge(element::Type& dst, const element::Type& t1, const element::Type& t2);

        private:
            Type_t m_type{Type_t::undefined};
        };

        extern NGRAPH_API const Type dynamic;
        extern NGRAPH_API const Type boolean;
        extern NGRAPH_API const Type bf16;
        extern NGRAPH_API const Type f32;
        extern NGRAPH_API const Type f64;
        extern NGRAPH_API const Type i8;
        extern NGRAPH_API const Type i16;
        extern NGRAPH_API const Type i32;
        extern NGRAPH_API const Type i64;
        extern NGRAPH_API const Type u8;
        extern NGRAPH_API const Type u16;
        extern NGRAPH_API const Type u32;
        extern NGRAPH_API const Type u64;

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
        template <>
        const Type& from<ngraph::bfloat16>();

        std::ostream& operator<<(std::ostream& out, const ngraph::element::Type& obj);
    }
}
