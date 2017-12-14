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
        /// \brief Abstract base class for constants.
        ///
        /// There are two subclasses: ParameterizedConstant and Constant. ParameterizedConstant allows constant values to be supplied via vectors of the corresponding C++ type;
        /// however, the ParameterizedConstant subclass can only be used when type information is available at C++ compile-time. In cases where types are not known until
        /// C++ runtime, the Constant subclass must be used instead.
        class ConstantBase : public Node
        {
        protected:
            /// \brief Constructs a constant base-type node.
            ///
            /// \param type The TensorViewType for the constant.
            ConstantBase(const std::string& node_type, const std::shared_ptr<TensorViewType>& type)
                : Node(node_type, {})
            {
                set_value_type_checked(type);
            }

            virtual bool is_constant() const override { return true; }
        };

        /// \brief Class for constants whose element types are known at C++ compile-time.
        ///
        /// \tparam T The ngraph::element::Type of the tensor's elements.
        ///
        /// This class can be used when the type of the tensor constant is known at C++ compile-time. For other cases, Constant must be used.
        ///
        /// ## Parameters
        ///
        /// |         | Description                                                                          |
        /// | ------- | ------------------------------------------------------------------------------------ |
        /// | `shape` | The ngraph::Shape of the tensor constant.                                            |
        /// | `value` | The ngraph::runtime::ParameterizedTensorView containing data fo the tensor constant. |
        ///
        /// ## Output
        ///
        /// | Type                   | Description                                                           |
        /// | ---------------------- | --------------------------------------------------------------------- |
        /// | \f$E[d_1,\dots,d_n]\f$ | A constant tensor with the specified element type, shape, and values. |
        template <typename T>
        class ParameterizedConstant : public ConstantBase
        {
        public:
            /// \brief The ngraph element type
            using element_type = T;
            /// \brief The C++ type that holds the element type
            using type = typename T::type;

            /// \brief Constructs a parameterized tensor constant.
            ///
            /// \param shape The shape of the tensor constant.
            /// \param value The value of the tensor constant.
            ParameterizedConstant(
                const Shape& shape,
                const typename std::shared_ptr<ngraph::runtime::ParameterizedTensorView<T>>& value)
                : ConstantBase("ParameterizedConstant",
                               std::make_shared<TensorViewType>(T::element_type(), shape))
                , m_value(value)
            {
            }

            virtual std::shared_ptr<Node> copy_with_new_args(
                const std::vector<std::shared_ptr<Node>>& new_args) const override
            {
                if (new_args.size() != 0)
                    throw ngraph_error("Incorrect number of new arguments");
                return std::make_shared<ParameterizedConstant<T>>(get_shape(), m_value);
            }

            /// \return The value of the tensor constant.
            typename std::shared_ptr<ngraph::runtime::ParameterizedTensorView<T>> get_value() const
            {
                return m_value;
            }

        protected:
            const std::shared_ptr<ngraph::runtime::ParameterizedTensorView<T>> m_value;
        };

        /// \brief A 32-bit floating-point tensor constant.
        using Float32Constant = ParameterizedConstant<element::Float32>;
        /// \brief A 64-bit floating-point tensor constant.
        using Float64Constant = ParameterizedConstant<element::Float64>;
        /// \brief An 8-bit signed integer tensor constant.
        using Int8Constant = ParameterizedConstant<element::Int8>;
        /// \brief A 16-bit signed integer tensor constant.
        using Int16Constant = ParameterizedConstant<element::Int16>;
        /// \brief A 32-bit signed integer tensor constant.
        using Int32Constant = ParameterizedConstant<element::Int32>;
        /// \brief A 64-bit signed integer tensor constant.
        using Int64Constant = ParameterizedConstant<element::Int64>;
        /// \brief An 8-bit unsigned integer tensor constant.
        using UInt8Constant = ParameterizedConstant<element::UInt8>;
        /// \brief A 16-bit unsigned integer tensor constant.
        using UInt16Constant = ParameterizedConstant<element::UInt16>;
        /// \brief A 32-bit unsigned integer tensor constant.
        using UInt32Constant = ParameterizedConstant<element::UInt32>;
        /// \brief A 64-bit unsigned integer tensor constant.
        using UInt64Constant = ParameterizedConstant<element::UInt64>;

        /// \brief Class for constants whose element types may not be known until graph construction time.
        ///
        /// This class must be used when the type of the tensor constant is unknown at C++ compile-time. For other cases, ParameterizedConstant should be used.
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

            static std::shared_ptr<op::Constant>
                create(const element::Type& et, Shape shape, std::initializer_list<double> values)
            {
                return std::make_shared<op::Constant>(et, shape, std::vector<double>{values});
            }

            static std::shared_ptr<op::Constant>
                create(const element::Type& et, Shape shape, std::initializer_list<int> values)
            {
                return std::make_shared<op::Constant>(et, shape, std::vector<int>{values});
            }

            static std::shared_ptr<op::Constant>
                create(const element::Type& et, Shape shape, std::initializer_list<size_t> values)
            {
                return std::make_shared<op::Constant>(et, shape, std::vector<size_t>{values});
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
