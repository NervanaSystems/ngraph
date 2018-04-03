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
                auto vt = std::make_shared<TensorViewType>(type, shape);
                set_value_type_checked(vt);
                if (values.size() == 1)
                {
                    write_values(std::vector<T>(shape_size(m_shape), values[0]));
                }
                else if (values.size() == shape_size(m_shape))
                {
                    write_values(values);
                }
                else
                {
                    throw ngraph_error("Constant does not have the expected number of literals");
                }
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
                auto vt = std::make_shared<TensorViewType>(type, shape);
                set_value_type_checked(vt);
                if (values.size() != shape_size(m_shape))
                {
                    throw ngraph_error("Constant does not have the expected number of literals");
                }
                std::vector<double> dvalues = parse_string<double>(values);
                write_values(dvalues);
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
                memcpy(m_data, data, size);
                auto vt = std::make_shared<TensorViewType>(type, shape);
                set_value_type_checked(vt);
            }

            virtual ~Constant() override;

            /// \brief Wrapper around constructing a shared_ptr of a Constant
            ///
            /// \param type The element type of the tensor constant.
            /// \param shape The shape of the tensor constant.
            /// \param values A vector of values to use as the constant data.
            template <typename T>
            static std::shared_ptr<op::Constant>
                create(const element::Type& type, Shape shape, const std::vector<T> values)
            {
                return std::make_shared<op::Constant>(type, shape, values);
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
                return std::make_shared<op::Constant>(type, shape, std::vector<T>{values});
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
            bool is_constant() const override { return true; }
        protected:
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
            Shape m_shape;
            void* m_data;
        };
    }
}
