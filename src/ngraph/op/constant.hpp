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

#include <cstring>
#include <sstream>

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/node.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
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
            NGRAPH_API
            static const std::string type_name;
            const std::string& description() const override { return type_name; }
            /// \brief Constructs a tensor constant.
            ///
            /// \param type The element type of the tensor constant.
            /// \param shape The shape of the tensor constant.
            /// \param values A vector of literals for initializing the tensor constant. The size
            ///        of values must match the size of the shape.
            template <typename T>
            Constant(const element::Type& type, Shape shape, const std::vector<T>& values)
                : m_element_type(type)
                , m_shape(shape)
                , m_data(new runtime::AlignedBuffer(shape_size(m_shape) * m_element_type.size(),
                                                    host_alignment()))
            {
                NODE_VALIDATION_CHECK(
                    this,
                    values.size() == 1 || values.size() == shape_size(m_shape),
                    "Did not get the expected number of literals for a constant of shape ",
                    m_shape,
                    " (got ",
                    values.size(),
                    ", expected ",
                    (shape_size(m_shape) == 1 ? "" : "1 or "),
                    shape_size(m_shape),
                    ").");

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
                : m_element_type(type)
                , m_shape(shape)
                , m_data(new runtime::AlignedBuffer(shape_size(m_shape) * m_element_type.size(),
                                                    host_alignment()))
            {
                NODE_VALIDATION_CHECK(
                    this,
                    values.size() == shape_size(m_shape) || values.size() == 1,
                    "Did not get the expected number of literals for a constant of shape ",
                    m_shape,
                    " (got ",
                    values.size(),
                    ", expected ",
                    shape_size(m_shape),
                    ".");
                if (values.size())
                {
                    if (type.is_integral())
                    {
                        if (type.is_signed())
                        {
                            std::vector<int64_t> dvalues = parse_string<int64_t>(values);
                            if (values.size() == 1 && shape_size(m_shape) != 1)
                            {
                                dvalues = std::vector<int64_t>(shape_size(m_shape), dvalues[0]);
                            }
                            write_values(dvalues);
                        }
                        else
                        {
                            std::vector<uint64_t> dvalues = parse_string<uint64_t>(values);
                            if (values.size() == 1 && shape_size(m_shape) != 1)
                            {
                                dvalues = std::vector<uint64_t>(shape_size(m_shape), dvalues[0]);
                            }
                            write_values(dvalues);
                        }
                    }
                    else
                    {
                        std::vector<double> dvalues = parse_string<double>(values);
                        if (values.size() == 1 && shape_size(m_shape) != 1)
                        {
                            dvalues = std::vector<double>(shape_size(m_shape), dvalues[0]);
                        }
                        write_values(dvalues);
                    }
                }
                constructor_validate_and_infer_types();
            }

            /// \brief Constructs a tensor constant with the same initialization value copied across
            //         the tensor. This constructor is to support deserialization of constants.
            ///
            /// \param type The element type of the tensor constant.
            /// \param shape The shape of the tensor constant.
            /// \param data A void* to constant data.
            Constant(const element::Type& type, const Shape& shape, const void* data)
                : m_element_type(type)
                , m_shape(shape)
                , m_data(nullptr)
            {
                size_t size = shape_size(m_shape) * m_element_type.size();
                m_data.reset(new runtime::AlignedBuffer(shape_size(m_shape) * m_element_type.size(),
                                                        host_alignment()));
                std::memcpy(m_data->get_ptr(), data, size);
                constructor_validate_and_infer_types();
            }

            virtual ~Constant() override;

            void validate_and_infer_types() override
            {
                infer_element_type();
                set_output_type(0, m_element_type, m_shape);
            }

            /// \brief Returns the value of the constant node as a Shape object
            ///        Can only be used on element::i64 nodes and interprets
            ///        negative values as zeros.
            Shape get_shape_val() const;
            /// \brief Returns the value of the constant node as a Strides
            ///        object
            ///        Can only be used on element::i64 nodes and interprets
            ///        negative values as zeros.
            Strides get_strides_val() const;
            /// \brief Returns the value of the constant node as a Coordinate
            ///        object
            ///        Can only be used on element::i64 nodes and interprets
            ///        negative values as zeros.
            Coordinate get_coordinate_val() const;
            /// \brief Returns the value of the constant node as a
            ///        CoordinateDiff object
            ///        Can only be used on element::i64 nodes.
            CoordinateDiff get_coordinate_diff_val() const;
            /// \brief Returns the value of the constant node as an AxisVector
            ///        object
            ///        Can only be used on element::i64 nodes and interprets
            ///        negative values as zeros.
            AxisVector get_axis_vector_val() const;
            /// \brief Returns the value of the constant node as an AxisSet
            ///        object
            ///        Can only be used on element::i64 nodes and interprets
            ///        negative values as zeros.
            ///        Repeated values are allowed.
            AxisSet get_axis_set_val() const;

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
                const T* p = reinterpret_cast<const T*>(m_data->get_ptr());
                for (size_t i = 0; i < shape_size(m_shape); i++)
                {
                    rc.push_back(p[i]);
                }
                return rc;
            }

            const void* get_data_ptr() const { return (m_data ? m_data->get_ptr() : nullptr); }
            template <typename T>
            const T* get_data_ptr() const
            {
                return reinterpret_cast<const T*>(get_data_ptr());
            }

            bool is_constant() const override { return true; }
            bool are_all_data_elements_bitwise_identical() const;
            std::string convert_value_to_string(size_t index) const;

        protected:
            void* get_data_ptr_nc() { return (m_data ? m_data->get_ptr() : nullptr); }
            Constant(const OutputVector& args)
                : Node(args)
                , m_shape({})
            {
            }

            virtual void infer_element_type() {}
            template <typename T>
            void write_values(const std::vector<T>& values)
            {
                write_to_buffer(
                    m_element_type, m_shape, values, get_data_ptr_nc(), shape_size(m_shape));
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
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif
                switch (target_type)
                {
                case element::Type_t::boolean:
                    write_buffer<char, T>(target, source, target_element_count);
                    break;
                case element::Type_t::bf16:
                    write_buffer<bfloat16, T>(target, source, target_element_count);
                    break;
                case element::Type_t::f16:
                    write_buffer<float16, T>(target, source, target_element_count);
                    break;
                case element::Type_t::f32:
                    write_buffer<float, T>(target, source, target_element_count);
                    break;
                case element::Type_t::f64:
                    write_buffer<double, T>(target, source, target_element_count);
                    break;
                case element::Type_t::i8:
                    write_buffer<int8_t, T>(target, source, target_element_count);
                    break;
                case element::Type_t::i16:
                    write_buffer<int16_t, T>(target, source, target_element_count);
                    break;
                case element::Type_t::i32:
                    write_buffer<int32_t, T>(target, source, target_element_count);
                    break;
                case element::Type_t::i64:
                    write_buffer<int64_t, T>(target, source, target_element_count);
                    break;
                case element::Type_t::u8:
                    write_buffer<uint8_t, T>(target, source, target_element_count);
                    break;
                case element::Type_t::u16:
                    write_buffer<uint16_t, T>(target, source, target_element_count);
                    break;
                case element::Type_t::u32:
                    write_buffer<uint32_t, T>(target, source, target_element_count);
                    break;
                case element::Type_t::u64:
                    write_buffer<uint64_t, T>(target, source, target_element_count);
                    break;
                case element::Type_t::undefined: throw std::runtime_error("unsupported type");
                case element::Type_t::dynamic: throw std::runtime_error("unsupported type");
                }
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif
            }

            static constexpr size_t host_alignment() { return 64; }
            element::Type m_element_type;
            Shape m_shape{};
            std::unique_ptr<runtime::AlignedBuffer> m_data;
            Constant(const Constant&) = delete;
            Constant operator=(const Constant&) = delete;
        };

        class ScalarConstantLikeBase : public Constant
        {
        public:
            NGRAPH_API
            static const std::string type_name;
            const std::string& description() const override { return type_name; }
            std::shared_ptr<op::Constant> as_constant() const;

        protected:
            ScalarConstantLikeBase(const OutputVector& args)
                : Constant(args)
            {
            }
        };

        /// \brief A scalar constant whose element type is the same as like.
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
            template <typename T>
            ScalarConstantLike(const Output<Node>& like, T value)
                : ScalarConstantLikeBase({like})
                , m_value(static_cast<double>(value))
            {
                constructor_validate_and_infer_types();
            }

            std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

        protected:
            void infer_element_type() override;

            double m_value;
        };
    }
}
