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

#include <onnx/onnx_pb.h>

#include "ngraph/except.hpp"
#include "tensor.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        // forward declarations
        class Graph;
        class Model;

        // Detecting automatically the underlying type used to store the information
        // for data type of values an attribute is holding. A bug was discovered in
        // protobuf which forced ONNX team to switch from `enum AttributeProto_AttributeType`
        // to `int32` in order to workaround the bug. This line allows using both versions
        // of ONNX generated wrappers.
        using AttributeProto_AttributeType = decltype(onnx::AttributeProto{}.type());

        namespace error
        {
            namespace attribute
            {
                namespace detail
                {
                    struct Attribute : ngraph_error
                    {
                        Attribute(const std::string& msg, AttributeProto_AttributeType type)
                            : ngraph_error{msg + ": " +
                                           onnx::AttributeProto_AttributeType_Name(type)}
                        {
                        }
                    };

                } // namespace detail

                struct InvalidData : detail::Attribute
                {
                    explicit InvalidData(AttributeProto_AttributeType type)
                        : Attribute{"invalid attribute type", type}
                    {
                    }
                };

                struct UnsupportedType : detail::Attribute
                {
                    explicit UnsupportedType(AttributeProto_AttributeType type)
                        : Attribute{"unsupported attribute type", type}
                    {
                    }
                };

            } // namespace attribute

        } // namespace error

        namespace detail
        {
            namespace attribute
            {
                template <typename T>
                inline T get_value(const onnx::AttributeProto& attribute)
                {
                    throw error::attribute::UnsupportedType{attribute.type()};
                }

                template <>
                inline float get_value(const onnx::AttributeProto& attribute)
                {
                    switch (attribute.type())
                    {
                    case onnx::AttributeProto_AttributeType_INT: return attribute.i();
                    case onnx::AttributeProto_AttributeType_FLOAT: return attribute.f();
                    default: throw error::attribute::InvalidData{attribute.type()};
                    }
                }

                template <>
                inline std::vector<float> get_value(const onnx::AttributeProto& attribute)
                {
                    switch (attribute.type())
                    {
                    case onnx::AttributeProto_AttributeType_INT:
                        return {static_cast<float>(attribute.i())};
                    case onnx::AttributeProto_AttributeType_INTS:
                        return {std::begin(attribute.floats()), std::end(attribute.floats())};
                    case onnx::AttributeProto_AttributeType_FLOAT: return {attribute.f()};
                    case onnx::AttributeProto_AttributeType_FLOATS:
                        return {std::begin(attribute.floats()), std::end(attribute.floats())};
                    default: throw error::attribute::InvalidData{attribute.type()};
                    }
                }

                template <>
                inline double get_value(const onnx::AttributeProto& attribute)
                {
                    switch (attribute.type())
                    {
                    case onnx::AttributeProto_AttributeType_FLOAT:
                        return static_cast<double>(attribute.f());
                    case onnx::AttributeProto_AttributeType_INT: return attribute.i();
                    default: throw error::attribute::InvalidData{attribute.type()};
                    }
                }

                template <>
                inline std::vector<double> get_value(const onnx::AttributeProto& attribute)
                {
                    switch (attribute.type())
                    {
                    case onnx::AttributeProto_AttributeType_INT:
                        return {static_cast<double>(attribute.i())};
                    case onnx::AttributeProto_AttributeType_INTS:
                        return {std::begin(attribute.ints()), std::end(attribute.ints())};
                    case onnx::AttributeProto_AttributeType_FLOAT:
                        return {static_cast<double>(attribute.f())};
                    case onnx::AttributeProto_AttributeType_FLOATS:
                        return {std::begin(attribute.floats()), std::end(attribute.floats())};
                    default: throw error::attribute::InvalidData{attribute.type()};
                    }
                }

                template <>
                inline std::size_t get_value(const onnx::AttributeProto& attribute)
                {
                    if (attribute.type() != onnx::AttributeProto_AttributeType_INT)
                    {
                        throw error::attribute::InvalidData{attribute.type()};
                    }
                    return static_cast<std::size_t>(attribute.i());
                }

                template <>
                inline std::vector<std::size_t> get_value(const onnx::AttributeProto& attribute)
                {
                    switch (attribute.type())
                    {
                    case onnx::AttributeProto_AttributeType_INT:
                        return {static_cast<std::size_t>(attribute.i())};
                    case onnx::AttributeProto_AttributeType_INTS:
                        return {std::begin(attribute.ints()), std::end(attribute.ints())};
                    default: throw error::attribute::InvalidData{attribute.type()};
                    }
                }

                template <>
                inline int64_t get_value(const onnx::AttributeProto& attribute)
                {
                    if (attribute.type() != onnx::AttributeProto_AttributeType_INT)
                    {
                        throw error::attribute::InvalidData{attribute.type()};
                    }
                    return attribute.i();
                }

                template <>
                inline std::vector<int64_t> get_value(const onnx::AttributeProto& attribute)
                {
                    switch (attribute.type())
                    {
                    case onnx::AttributeProto_AttributeType_INT: return {attribute.i()};
                    case onnx::AttributeProto_AttributeType_INTS:
                        return {std::begin(attribute.ints()), std::end(attribute.ints())};
                    default: throw error::attribute::InvalidData{attribute.type()};
                    }
                }

                template <>
                inline std::string get_value(const onnx::AttributeProto& attribute)
                {
                    if (attribute.type() != onnx::AttributeProto_AttributeType_STRING)
                    {
                        throw error::attribute::InvalidData{attribute.type()};
                    }
                    return attribute.s();
                }

                template <>
                inline std::vector<std::string> get_value(const onnx::AttributeProto& attribute)
                {
                    switch (attribute.type())
                    {
                    case onnx::AttributeProto_AttributeType_STRING: return {attribute.s()};
                    case onnx::AttributeProto_AttributeType_STRINGS:
                        return {std::begin(attribute.strings()), std::end(attribute.strings())};
                    default: throw error::attribute::InvalidData{attribute.type()};
                    }
                }

                template <>
                inline Tensor get_value(const onnx::AttributeProto& attribute)
                {
                    if (attribute.type() != onnx::AttributeProto_AttributeType_TENSOR)
                    {
                        throw error::attribute::InvalidData{attribute.type()};
                    }
                    return Tensor{attribute.t()};
                }

                template <>
                inline std::vector<Tensor> get_value(const onnx::AttributeProto& attribute)
                {
                    switch (attribute.type())
                    {
                    case onnx::AttributeProto_AttributeType_TENSOR: return {Tensor{attribute.t()}};
                    case onnx::AttributeProto_AttributeType_TENSORS:
                        return {std::begin(attribute.tensors()), std::end(attribute.tensors())};
                    default: throw error::attribute::InvalidData{attribute.type()};
                    }
                }

            } // namespace attribute

        } // namespace detail

        class Attribute
        {
        public:
            enum class Type
            {
                undefined = onnx::AttributeProto_AttributeType_UNDEFINED,
                float_point = onnx::AttributeProto_AttributeType_FLOAT,
                integer = onnx::AttributeProto_AttributeType_INT,
                string = onnx::AttributeProto_AttributeType_STRING,
                tensor = onnx::AttributeProto_AttributeType_TENSOR,
                graph = onnx::AttributeProto_AttributeType_GRAPH,
                float_point_array = onnx::AttributeProto_AttributeType_FLOATS,
                integer_array = onnx::AttributeProto_AttributeType_INTS,
                string_array = onnx::AttributeProto_AttributeType_STRINGS,
                tensor_array = onnx::AttributeProto_AttributeType_TENSORS,
                graph_array = onnx::AttributeProto_AttributeType_GRAPHS
            };

            Attribute() = delete;
            explicit Attribute(const onnx::AttributeProto& attribute_proto)
                : m_attribute_proto{&attribute_proto}
            {
            }

            Attribute(Attribute&&) noexcept = default;
            Attribute(const Attribute&) = default;

            Attribute& operator=(Attribute&&) noexcept = delete;
            Attribute& operator=(const Attribute&) = delete;

            const std::string& get_name() const { return m_attribute_proto->name(); }
            Type get_type() const { return static_cast<Type>(m_attribute_proto->type()); }
            bool is_tensor() const { return get_type() == Type::tensor; }
            bool is_tensor_array() const { return get_type() == Type::tensor_array; }
            bool is_float() const { return get_type() == Type::float_point; }
            bool is_float_array() const { return get_type() == Type::float_point_array; }
            bool is_integer() const { return get_type() == Type::integer; }
            bool is_integer_array() const { return get_type() == Type::integer_array; }
            bool is_string() const { return get_type() == Type::string; }
            bool is_string_array() const { return get_type() == Type::string_array; }
            bool is_graph() const { return get_type() == Type::graph; }
            bool is_graph_array() const { return get_type() == Type::graph_array; }
            Tensor get_tensor() const { return Tensor{m_attribute_proto->t()}; }
            float get_float() const { return m_attribute_proto->f(); }
            int64_t get_integer() const { return m_attribute_proto->i(); }
            const std::string& get_string() const { return m_attribute_proto->s(); }
            Graph get_graph(Model&) const;

            std::vector<Tensor> get_tensor_array() const
            {
                return {std::begin(m_attribute_proto->tensors()),
                        std::end(m_attribute_proto->tensors())};
            }

            std::vector<float> get_float_array() const
            {
                return {std::begin(m_attribute_proto->floats()),
                        std::end(m_attribute_proto->floats())};
            }

            std::vector<int64_t> get_integer_array() const
            {
                return {std::begin(m_attribute_proto->ints()), std::end(m_attribute_proto->ints())};
            }

            std::vector<std::string> get_string_array() const
            {
                return {std::begin(m_attribute_proto->strings()),
                        std::end(m_attribute_proto->strings())};
            }

            std::vector<Graph> get_graph_array(Model&) const;

            /* explicit */ operator onnx::AttributeProto_AttributeType() const
            {
                return m_attribute_proto->type();
            }

            template <typename T>
            T get_value() const
            {
                return detail::attribute::get_value<T>(*m_attribute_proto);
            }

        private:
            const onnx::AttributeProto* m_attribute_proto;
        };

    } // namespace onnx_import

} // namespace ngraph
