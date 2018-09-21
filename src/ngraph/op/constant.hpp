//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include <cstring>
#include <sstream>

#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Class for constants.
        class Constant : public Node
        {
        public:
            /// \brief Constructs a tensor constant.
            ///
            /// \param type The element type of the tensor constant.
            /// \param shape The shape of the tensor constant.
            /// \param values A vector of literals for initializing the tensor constant. The size
            ///        of values must match the size of the shape.
            template <typename T>
            Constant(const element::Type& type, Shape shape, const std::vector<T>& values)
                : Node("Constant", {})
                , m_element_type(type)
                , m_shape(shape)
                , m_data(ngraph::aligned_alloc(m_element_type.size(),
                                               shape_size(m_shape) * m_element_type.size()))
            {
                NODE_VALIDATION_ASSERT(this,
                                       values.size() == 1 || values.size() == shape_size(m_shape))
                    << "Did not get the expected number of literals for a constant of shape "
                    << m_shape << " (got " << values.size() << ", expected "
                    << (shape_size(m_shape) == 1 ? "" : "1 or ") << shape_size(m_shape) << ").";

                if (values.size() == 1)
                {
                    write_values(std::vector<T>(shape_size(m_shape), values[0]));
                }
                else
                {
                    write_values(values);
                }
                constructor_validate_and_infer_types();
            }

            /// \brief Constructs a tensor constant
            ///        This constructor is mainly to support deserialization of constants.
            ///
            /// \param type The element type of the tensor constant.
            /// \param shape The shape of the tensor constant.
            /// \param values A list of string values to use as the constant data.
            Constant(const element::Type& type, Shape shape, const std::vector<std::string>& values)
                : Node("Constant", {})
                , m_element_type(type)
                , m_shape(shape)
                , m_data(ngraph::aligned_alloc(m_element_type.size(),
                                               shape_size(m_shape) * m_element_type.size()))
            {
                NODE_VALIDATION_ASSERT(this, values.size() == shape_size(m_shape))
                    << "Did not get the expected number of literals for a constant of shape "
                    << m_shape << " (got " << values.size() << ", expected " << shape_size(m_shape)
                    << ".";

                std::vector<double> dvalues = parse_string<double>(values);
                write_values(dvalues);
                constructor_validate_and_infer_types();
            }

            /// \brief Constructs a tensor constant with the same initialization value copied across
            //         the tensor. This constructor is to support deserialization of constants.
            ///
            /// \param type The element type of the tensor constant.
            /// \param shape The shape of the tensor constant.
            /// \param data A void* to constant data.
            Constant(const element::Type& type, const Shape& shape, const void* data)
                : Node("Constant", {})
                , m_element_type(type)
                , m_shape(shape)
                , m_data(nullptr)
            {
                size_t size = shape_size(m_shape) * m_element_type.size();
                m_data = ngraph::aligned_alloc(m_element_type.size(), size);
                std::memcpy(m_data, data, size);
                constructor_validate_and_infer_types();
            }

            virtual ~Constant() override;

            void validate_and_infer_types() override
            {
                infer_element_type();
                set_output_type(0, m_element_type, m_shape);
                if (m_element_type == element::i64)
                {
                    set_output_static_value(0, get_vector<size_t>());
                }
            }

            /// \brief Wrapper around constructing a shared_ptr of a Constant
            ///
            /// \param type The element type of the tensor constant.
            /// \param shape The shape of the tensor constant.
            /// \param values A vector of values to use as the constant data.
            template <typename T>
            static std::shared_ptr<op::Constant>
                create(const element::Type& type, Shape shape, const std::vector<T> values)
            {
                auto result = std::make_shared<op::Constant>(type, shape, values);
                result->validate_and_infer_types();
                return result;
            }

            /// \brief Wrapper around constructing a shared_ptr of a Constant
            ///
            /// \param type The element type of the tensor constant.
            /// \param shape The shape of the tensor constant.
            /// \param values An initializer_list of values to use as the constant data.
            template <typename T>
            static std::shared_ptr<op::Constant>
                create(const element::Type& type, Shape shape, std::initializer_list<T> values)
            {
                auto result = std::make_shared<op::Constant>(type, shape, std::vector<T>{values});
                result->validate_and_infer_types();
                return result;
            }

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            /// \return The initialization literals for the tensor constant.
            std::vector<std::string> get_value_strings() const;

            template <typename T>
            std::vector<T> get_vector() const
            {
                if (sizeof(T) > m_element_type.size() && shape_size(m_shape) > 0)
                {
                    throw ngraph_error("Buffer over-read");
                }

                std::vector<T> rc;
                const T* p = reinterpret_cast<const T*>(m_data);
                for (size_t i = 0; i < shape_size(m_shape); i++)
                {
                    rc.push_back(p[i]);
                }
                return rc;
            }

            const void* get_data_ptr() const { return m_data; }
            template <typename T>
            const T* get_data_ptr() const
            {
                return reinterpret_cast<T*>(m_data);
            }

            bool is_constant() const override { return true; }
        protected:
            Constant(const std::string& name, const NodeVector& args)
                : Node(name, args)
                , m_shape({})
            {
            }

            virtual void infer_element_type() {}
            template <typename T>
            void write_values(const std::vector<T>& values)
            {
                write_to_buffer(m_element_type, m_shape, values, m_data, shape_size(m_shape));
            }

            template <typename T, typename U>
            void write_buffer(void* target, const std::vector<U>& source, size_t count)
            {
                T* p = reinterpret_cast<T*>(target);
                for (size_t i = 0; i < count; i++)
                {
                    p[i] = static_cast<T>(source[i]);
                }
            }

            template <typename T>
            void write_to_buffer(const element::Type& target_type,
                                 const Shape& target_shape,
                                 const std::vector<T>& source,
                                 void* target,
                                 size_t target_element_count)
            {
                if (source.size() != target_element_count)
                {
                    throw std::runtime_error("Constant initializer does not match shape");
                }
                if (target_type == element::boolean)
                {
                    write_buffer<char, T>(target, source, target_element_count);
                }
                else if (target_type == element::f32)
                {
                    write_buffer<float, T>(target, source, target_element_count);
                }
                else if (target_type == element::f64)
                {
                    write_buffer<double, T>(target, source, target_element_count);
                }
                else if (target_type == element::i8)
                {
                    write_buffer<int8_t, T>(target, source, target_element_count);
                }
                else if (target_type == element::i16)
                {
                    write_buffer<int16_t, T>(target, source, target_element_count);
                }
                else if (target_type == element::i32)
                {
                    write_buffer<int32_t, T>(target, source, target_element_count);
                }
                else if (target_type == element::i64)
                {
                    write_buffer<int64_t, T>(target, source, target_element_count);
                }
                else if (target_type == element::u8)
                {
                    write_buffer<uint8_t, T>(target, source, target_element_count);
                }
                else if (target_type == element::u16)
                {
                    write_buffer<uint16_t, T>(target, source, target_element_count);
                }
                else if (target_type == element::u32)
                {
                    write_buffer<uint32_t, T>(target, source, target_element_count);
                }
                else if (target_type == element::u64)
                {
                    write_buffer<uint64_t, T>(target, source, target_element_count);
                }
                else
                {
                    throw std::runtime_error("unsupported type");
                }
            }

            element::Type m_element_type;
            Shape m_shape{};
            void* m_data{nullptr};
            Constant(const Constant&) = delete;
            Constant(Constant&&) = delete;
            Constant operator=(const Constant*) = delete;
        };

        class ScalarConstantLikeBase : public Constant
        {
        public:
            std::shared_ptr<op::Constant> as_constant() const;

        protected:
            ScalarConstantLikeBase(const std::string& name, const NodeVector& args)
                : Constant(name, args)
            {
            }
        };

        /// \brief A scalar constant whose element type is the same as like.
        template <typename T>
        class ScalarConstantLike : public ScalarConstantLikeBase
        {
        public:
            /// \brief A scalar constant whose element type is the same as like.
            ///
            /// Once the element type is known, the dependency on like will be removed and
            /// this node will be replaced with an equivalent constant.
            ///
            /// \param like A tensor that will supply the element type.
            /// \param value The value of the scalar.
            ScalarConstantLike(const std::shared_ptr<Node>& like, T value)
                : ScalarConstantLikeBase("ScalarConstantLike", {like})
                , m_value(value)
            {
                constructor_validate_and_infer_types();
            }

            std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override
            {
                return std::make_shared<ScalarConstantLike<T>>(new_args.at(0), m_value);
            }

        protected:
            void infer_element_type() override
            {
                m_element_type = get_input_element_type(0);
                if (nullptr == m_data)
                {
                    m_data = ngraph::aligned_alloc(m_element_type.size(), m_element_type.size());
                    write_values(std::vector<T>(1, m_value));
                }
            }

            T m_value;
        };
    }
}
