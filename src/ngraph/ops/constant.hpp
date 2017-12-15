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

#pragma once

#include <sstream>

#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "ngraph/runtime/utils.hpp"
#include "ngraph/types/element_type.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Class for constants.
        ///
        /// ## Parameters
        ///
        /// |                 | Description                                                                                                                                                                    |
        /// | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
        /// | `et`            | The ngraph::element::Type of the tensor constant.                                                                                                                              |
        /// | `shape`         | The ngraph::Shape of the tensor constant.                                                                                                                                      |
        /// | `value_strings` | A list of strings containing literals for initialization of the tensor constant. These strings are parsed with the appropriate instance of ngraph::element::TraitedType::read. |
        ///
        /// ## Output
        ///
        /// | Type                   | Description                                                           |
        /// | ---------------------- | --------------------------------------------------------------------- |
        /// | \f$E[d_1,\dots,d_n]\f$ | A constant tensor with the specified element type, shape, and values. |
        class Constant : public Node
        {
        public:
            /// \brief Constructs a tensor constant.
            ///
            /// \param shape The shape of the tensor constant.
            /// \param values A list of literals for initializing the tensor constant. There must be one literal for each element of the tensor; i.e., `value_strings.size()` must equal `ngraph::shape_size(shape)`.
            template <typename T>
            Constant(const element::Type& et, Shape shape, const std::vector<T>& values)
                : Node("Constant", {})
                , m_element_type(et)
                , m_shape(shape)
                , m_data(aligned_alloc(m_element_type.size(),
                                       shape_size(m_shape) * m_element_type.size()))
            {
                auto vt = std::make_shared<TensorViewType>(et, shape);
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

            Constant(const element::Type& et, Shape shape, const std::vector<std::string>& values)
                : Node("Constant", {})
                , m_element_type(et)
                , m_shape(shape)
                , m_data(aligned_alloc(m_element_type.size(),
                                       shape_size(m_shape) * m_element_type.size()))
            {
                auto vt = std::make_shared<TensorViewType>(et, shape);
                set_value_type_checked(vt);
                if (values.size() != 1 && values.size() != shape_size(m_shape))
                {
                    throw ngraph_error("Constant does not have the expected number of literals");
                }
                // write_values(values);
                std::vector<double> dvalues = parse_string<double>(values);
                if (dvalues.size() == 1)
                {
                    dvalues = std::vector<double>(shape_size(m_shape), dvalues[0]);
                }
                write_values(dvalues);
            }

            Constant(const element::Type& et, const Shape& shape, const void* data)
                : Node("Constant", {})
                , m_element_type(et)
                , m_shape(shape)
                , m_data(nullptr)
            {
                size_t size = shape_size(m_shape) * m_element_type.size();
                m_data = aligned_alloc(m_element_type.size(), size);
                memcpy(m_data, data, size);
                auto vt = std::make_shared<TensorViewType>(et, shape);
                set_value_type_checked(vt);
            }

            virtual ~Constant();

            template <typename T>
            static std::shared_ptr<op::Constant>
                create(const element::Type& et, Shape shape, const std::vector<T> values)
            {
                return std::make_shared<op::Constant>(et, shape, values);
            }

            template <typename T>
            static std::shared_ptr<op::Constant>
                create(const element::Type& et, Shape shape, std::initializer_list<T> values)
            {
                return std::make_shared<op::Constant>(et, shape, std::vector<T>{values});
            }

            // /// \brief Constructs a tensor constant with the same initialization value copied across the tensor.
            // ///
            // /// \param et The element type of the tensor constant.
            // /// \param shape The shape of the tensor constant.
            // /// \param value_string A literal for initializing each tensor constant.
            // Constant(const element::Type& et, const Shape& shape, const std::string& value_string);

            virtual std::shared_ptr<Node> copy_with_new_args(
                const std::vector<std::shared_ptr<Node>>& new_args) const override
            {
                if (new_args.size() != 0)
                {
                    throw ngraph_error("Incorrect number of new arguments");
                }
                return std::make_shared<Constant>(m_element_type, m_shape, m_data);
            }

            /// \return The initialization literals for the tensor constant.
            std::vector<std::string> get_value_strings() const;

            template <typename T>
            std::vector<T> get_vector() const
            {
                std::vector<T> rc;
                const T* p = reinterpret_cast<const T*>(m_data);
                for (size_t i = 0; i < shape_size(m_shape); i++)
                {
                    rc.push_back(p[i]);
                }
                return rc;
            }

            void* get_data_ptr() { return m_data; }
            virtual bool is_constant() const override { return true; }
        protected:
            // void check_args();
            template <typename T>
            void write_values(const std::vector<T>& values)
            {
                write_to_buffer(m_element_type, m_shape, values, m_data, shape_size(m_shape));
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
                for (size_t i = 0; i < target_element_count; i++)
                {
                    if (target_type == element::boolean)
                    {
                        char* p = reinterpret_cast<char*>(target);
                        p[i] = static_cast<char>(source[i]);
                    }
                    else if (target_type == element::f32)
                    {
                        float* p = reinterpret_cast<float*>(target);
                        float tmp = static_cast<float>(source[i]);
                        p[i] = tmp;
                    }
                    else if (target_type == element::f64)
                    {
                        double* p = reinterpret_cast<double*>(target);
                        p[i] = static_cast<double>(source[i]);
                    }
                    else if (target_type == element::i8)
                    {
                        int8_t* p = reinterpret_cast<int8_t*>(target);
                        p[i] = static_cast<int8_t>(source[i]);
                    }
                    else if (target_type == element::i16)
                    {
                        int16_t* p = reinterpret_cast<int16_t*>(target);
                        p[i] = static_cast<int16_t>(source[i]);
                    }
                    else if (target_type == element::i32)
                    {
                        int32_t* p = reinterpret_cast<int32_t*>(target);
                        p[i] = static_cast<int32_t>(source[i]);
                    }
                    else if (target_type == element::i64)
                    {
                        int64_t* p = reinterpret_cast<int64_t*>(target);
                        p[i] = static_cast<int64_t>(source[i]);
                    }
                    else if (target_type == element::u8)
                    {
                        uint8_t* p = reinterpret_cast<uint8_t*>(target);
                        p[i] = static_cast<uint8_t>(source[i]);
                    }
                    else if (target_type == element::u16)
                    {
                        uint16_t* p = reinterpret_cast<uint16_t*>(target);
                        p[i] = static_cast<uint16_t>(source[i]);
                    }
                    else if (target_type == element::u32)
                    {
                        uint32_t* p = reinterpret_cast<uint32_t*>(target);
                        p[i] = static_cast<uint32_t>(source[i]);
                    }
                    else if (target_type == element::u64)
                    {
                        uint64_t* p = reinterpret_cast<uint64_t*>(target);
                        p[i] = static_cast<uint64_t>(source[i]);
                    }
                    else
                    {
                        throw std::runtime_error("unsupported type");
                    }
                }
            }

            element::Type m_element_type;
            Shape m_shape;
            void* m_data;
        };
    }
}
