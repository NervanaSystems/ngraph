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

#pragma once

#include "ngraph/except.hpp"
#include "onnx.pb.h"
#include "tensor.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        // forward declaration
        class Graph;

        namespace error
        {
            namespace attribute
            {
                namespace detail
                {
                    struct attribute : ngraph_error
                    {
                        attribute(std::string msg, onnx::AttributeProto_AttributeType type)
                            : ngraph_error{std::move(msg) + ": " +
                                           onnx::AttributeProto_AttributeType_Name(type)}
                        {
                        }
                    };

                } // namespace detail

                struct invalid_data : detail::attribute
                {
                    explicit invalid_data(onnx::AttributeProto_AttributeType type)
                        : attribute{"invalid attribute type", type}
                    {
                    }
                };

                struct unsupported_type : detail::attribute
                {
                    explicit unsupported_type(onnx::AttributeProto_AttributeType type)
                        : attribute("unsupported attribute type", type)
                    {
                    }
                };

            } // namespace attribute

        } // namespace error

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
                : m_attribute_proto{attribute_proto}
            {
            }

            Attribute(Attribute&&) noexcept = default;
            Attribute(const Attribute&) = default;

            Attribute& operator=(Attribute&&) noexcept = delete;
            Attribute& operator=(const Attribute&) = delete;

            const std::string& get_name() const { return m_attribute_proto.name(); }
            Type get_type() const { return static_cast<Type>(m_attribute_proto.type()); }
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
            Tensor get_tensor() const { return Tensor{m_attribute_proto.t()}; }
            float get_float() const { return m_attribute_proto.f(); }
            int64_t get_integer() const { return m_attribute_proto.i(); }
            const std::string& get_string() const { return m_attribute_proto.s(); }
            Graph get_graph() const;

            std::vector<Tensor> get_tensor_array() const
            {
                return {std::begin(m_attribute_proto.tensors()),
                        std::end(m_attribute_proto.tensors())};
            }

            std::vector<float> get_float_array() const
            {
                return {std::begin(m_attribute_proto.floats()),
                        std::end(m_attribute_proto.floats())};
            }

            std::vector<int64_t> get_integer_array() const
            {
                return {std::begin(m_attribute_proto.ints()), std::end(m_attribute_proto.ints())};
            }

            std::vector<std::string> get_string_array() const
            {
                return {std::begin(m_attribute_proto.strings()),
                        std::end(m_attribute_proto.strings())};
            }

            std::vector<Graph> get_graph_array() const;

            /* explicit */ operator onnx::AttributeProto_AttributeType() const
            {
                return m_attribute_proto.type();
            }

        private:
            const onnx::AttributeProto& m_attribute_proto;
        };

    } // namespace onnx_import

} // namespace ngraph
