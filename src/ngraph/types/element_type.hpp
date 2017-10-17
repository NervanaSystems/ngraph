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
#include <string>
#include <type_traits>

#include "ngraph/common.hpp"
#include "ngraph/except.hpp"
#include "ngraph/runtime/parameterized_tensor_view.hpp"
#include "ngraph/runtime/tensor_view.hpp"

namespace ngraph
{
    namespace element
    {
        class Type
        {
            Type(const Type&) = delete;
            Type& operator=(const Type&) = delete;

        public:
            virtual ~Type() {}
            Type(size_t bitwidth, bool is_float, bool is_signed, const std::string& cname);

            const std::string& c_type_string() const;
            size_t size() const;
            size_t hash() const
            {
                std::hash<std::string> h;
                return h(m_cname);
            }

            virtual std::shared_ptr<ngraph::runtime::TensorView>
                make_primary_tensor_view(const Shape& shape) const = 0;

            bool operator==(const Type& other) const;
            bool operator!=(const Type& other) const { return !(*this == other); }
            friend std::ostream& operator<<(std::ostream&, const Type&);

        private:
            static std::map<std::string, Type> m_element_list;
            size_t m_bitwidth;
            bool m_is_float;
            bool m_is_signed;
            const std::string m_cname;
        };

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

            virtual std::shared_ptr<ngraph::runtime::TensorView>
                make_primary_tensor_view(const ngraph::Shape& shape) const override
            {
                return std::make_shared<runtime::ParameterizedTensorView<TraitedType<T>>>(shape);
            }

            /// Parses a string containing a literal of the underlying type.
            static T read(const std::string& s)
            {
                T result;
                std::stringstream ss;

                ss << s;
                ss >> result;

                // Check that (1) parsing succeeded and (2) the entire string was used.
                if (ss.fail() || ss.rdbuf()->in_avail() != 0)
                {
                    throw ngraph_error("Could not parse literal");
                }

                return result;
            }

            /// Parses a list of strings containing literals of the underlying type.
            static std::vector<T> read(const std::vector<std::string>& ss)
            {
                std::vector<T> result;

                for (auto s : ss)
                {
                    result.push_back(read(s));
                }

                return result;
            }
        };

        NGRAPH_DEFINE_TRAITED_TYPE_NAME(char)
        using Bool = TraitedType<char>;

        NGRAPH_DEFINE_TRAITED_TYPE_NAME(float)
        using Float32 = TraitedType<float>;

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

//
// Utility macro for dispatching an element type-templated function at runtime.
//

// clang-format off
// Sorry, but you really don't want to see what clang-format does to this thing. :)
#define FUNCTION_ON_ELEMENT_TYPE(et, err_msg, f, ...)                                     \
    (                                                                                     \
        ((et) == element::Bool::element_type()) ? (f<element::Bool>(__VA_ARGS__)) :       \
        ((et) == element::Float32::element_type()) ? (f<element::Float32>(__VA_ARGS__)) : \
        ((et) == element::Int8::element_type()) ? (f<element::Int8>(__VA_ARGS__)) :       \
        ((et) == element::Int32::element_type()) ? (f<element::Int32>(__VA_ARGS__)) :     \
        ((et) == element::Int64::element_type()) ? (f<element::Int64>(__VA_ARGS__)) :     \
        ((et) == element::UInt8::element_type()) ? (f<element::UInt8>(__VA_ARGS__)) :     \
        ((et) == element::UInt32::element_type()) ? (f<element::UInt32>(__VA_ARGS__)) :   \
        ((et) == element::UInt64::element_type()) ? (f<element::UInt64>(__VA_ARGS__)) :   \
        (throw ngraph_error(err_msg))                                                     \
    )
// clang-format on
