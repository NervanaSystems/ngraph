//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <cmath>
#include <cstring>
#include <sstream>

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/node.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/type/element_type_traits.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            /// \brief Class for constants.
            class NGRAPH_API Constant : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"Constant", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                Constant() = default;

                /// \brief Initialize a constant from tensor
                /// \param tensor The tensor with data
                Constant(const std::shared_ptr<runtime::Tensor>& tensor);

                /// \brief Constructs a tensor constant.
                ///
                /// \param type The element type of the tensor constant.
                /// \param shape The shape of the tensor constant.
                /// \param values A vector of literals for initializing the tensor constant. The
                ///               size of values must match the size of the shape.
                template <typename T>
                Constant(const element::Type& type, Shape shape, const std::vector<T>& values)
                    : Constant(type, shape)
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
                    m_all_elements_bitwise_identical = are_all_data_elements_bitwise_identical();
                }

                /// \brief Create unitialized constant
                Constant(const element::Type& type, const Shape& shape);
                /// \brief Constructs a uniform tensor constant.
                ///
                /// \param type The element type of the tensor constant.
                /// \param shape The shape of the tensor constant.
                /// \param value A scalar for initializing the uniform tensor constant. The
                ///               value is broadcast to the specified shape.
                template <class T,
                          class = typename std::enable_if<std::is_fundamental<T>::value>::type>
                Constant(const element::Type& type, Shape shape, T value)
                    : Constant(type, shape)
                {
                    auto size = shape_size(m_shape);
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif
                    switch (type)
                    {
                    case element::Type_t::boolean:
                        std::fill_n(
                            get_data_ptr_nc<element::Type_t::boolean>(),
                            size,
                            static_cast<
                                typename element_type_traits<element::Type_t::boolean>::value_type>(
                                value));
                        break;
                    case element::Type_t::bf16:
                        std::fill_n(
                            get_data_ptr_nc<element::Type_t::bf16>(),
                            size,
                            static_cast<
                                typename element_type_traits<element::Type_t::bf16>::value_type>(
                                value));
                        break;
                    case element::Type_t::f16:
                        std::fill_n(
                            get_data_ptr_nc<element::Type_t::f16>(),
                            size,
                            static_cast<
                                typename element_type_traits<element::Type_t::f16>::value_type>(
                                value));
                        break;
                    case element::Type_t::f32:
                        std::fill_n(
                            get_data_ptr_nc<element::Type_t::f32>(),
                            size,
                            static_cast<
                                typename element_type_traits<element::Type_t::f32>::value_type>(
                                value));
                        break;
                    case element::Type_t::f64:
                        std::fill_n(
                            get_data_ptr_nc<element::Type_t::f64>(),
                            size,
                            static_cast<
                                typename element_type_traits<element::Type_t::f64>::value_type>(
                                value));
                        break;
                    case element::Type_t::i8:
                        std::fill_n(
                            get_data_ptr_nc<element::Type_t::i8>(),
                            size,
                            static_cast<
                                typename element_type_traits<element::Type_t::i8>::value_type>(
                                value));
                        break;
                    case element::Type_t::i16:
                        std::fill_n(
                            get_data_ptr_nc<element::Type_t::i16>(),
                            size,
                            static_cast<
                                typename element_type_traits<element::Type_t::i16>::value_type>(
                                value));
                        break;
                    case element::Type_t::i32:
                        std::fill_n(
                            get_data_ptr_nc<element::Type_t::i32>(),
                            size,
                            static_cast<
                                typename element_type_traits<element::Type_t::i32>::value_type>(
                                value));
                        break;
                    case element::Type_t::i64:
                        std::fill_n(
                            get_data_ptr_nc<element::Type_t::i64>(),
                            size,
                            static_cast<
                                typename element_type_traits<element::Type_t::i64>::value_type>(
                                value));
                        break;
                    case element::Type_t::u8:
                        std::fill_n(
                            get_data_ptr_nc<element::Type_t::u8>(),
                            size,
                            static_cast<
                                typename element_type_traits<element::Type_t::u8>::value_type>(
                                value));
                        break;
                    case element::Type_t::u16:
                        std::fill_n(
                            get_data_ptr_nc<element::Type_t::u16>(),
                            size,
                            static_cast<
                                typename element_type_traits<element::Type_t::u16>::value_type>(
                                value));
                        break;
                    case element::Type_t::u32:
                        std::fill_n(
                            get_data_ptr_nc<element::Type_t::u32>(),
                            size,
                            static_cast<
                                typename element_type_traits<element::Type_t::u32>::value_type>(
                                value));
                        break;
                    case element::Type_t::u64:
                        std::fill_n(
                            get_data_ptr_nc<element::Type_t::u64>(),
                            size,
                            static_cast<
                                typename element_type_traits<element::Type_t::u64>::value_type>(
                                value));
                        break;
                    case element::Type_t::u1: throw std::runtime_error("unsupported type");
                    case element::Type_t::undefined: throw std::runtime_error("unsupported type");
                    case element::Type_t::dynamic: throw std::runtime_error("unsupported type");
                    }
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif
                    constructor_validate_and_infer_types();
                    m_all_elements_bitwise_identical = true;
                }

                /// \brief Constructs a tensor constant
                ///        This constructor is mainly to support deserialization of constants.
                ///
                /// \param type The element type of the tensor constant.
                /// \param shape The shape of the tensor constant.
                /// \param values A list of string values to use as the constant data.
                Constant(const element::Type& type,
                         Shape shape,
                         const std::vector<std::string>& values);

                /// \brief Constructs a tensor constant with the supplied data
                ///
                /// \param type The element type of the tensor constant.
                /// \param shape The shape of the tensor constant.
                /// \param data A void* to constant data.
                Constant(const element::Type& type, const Shape& shape, const void* data);

                Constant(const Constant& other);
                Constant& operator=(const Constant&) = delete;

                virtual ~Constant() override;

                void validate_and_infer_types() override
                {
                    infer_element_type();
                    set_output_type(0, m_element_type, m_shape);
                }

                bool visit_attributes(AttributeVisitor& visitor) override;

                bool evaluate(const HostTensorVector& outputs,
                              const HostTensorVector& inputs) const override;

                // Don't constant fold a constant; it would make a copy
                bool constant_fold(OutputVector& outputs, const OutputVector& inputs) override
                {
                    return false;
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
                static std::shared_ptr<op::v0::Constant>
                    create(const element::Type& type, Shape shape, const std::vector<T> values)
                {
                    auto result = std::make_shared<op::v0::Constant>(type, shape, values);
                    result->validate_and_infer_types();
                    return result;
                }

                /// \brief Wrapper around constructing a shared_ptr of a Constant
                ///
                /// \param type The element type of the tensor constant.
                /// \param shape The shape of the tensor constant.
                /// \param values An initializer_list of values to use as the constant data.
                template <typename T>
                static std::shared_ptr<op::v0::Constant>
                    create(const element::Type& type, Shape shape, std::initializer_list<T> values)
                {
                    auto result =
                        std::make_shared<op::v0::Constant>(type, shape, std::vector<T>{values});
                    result->validate_and_infer_types();
                    return result;
                }

                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;

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
                    const T* p = static_cast<const T*>(get_data_ptr());
                    for (size_t i = 0; i < shape_size(m_shape); i++)
                    {
                        rc.push_back(p[i]);
                    }
                    return rc;
                }

                /// \brief Return the Constant's value as a vector cast to type T
                ///
                /// \tparam T  Type to which data vector's entries will be cast.
                /// \return    Constant's data vector.
                template <typename T>
                std::vector<T> cast_vector() const
                {
                    auto source_type = get_output_element_type(0);
                    std::vector<T> rc;
                    switch (source_type)
                    {
                    case element::Type_t::boolean:
                    {
                        auto vector = get_vector<char>();
                        rc = std::vector<T>(vector.begin(), vector.end());
                        break;
                    }
                    case element::Type_t::bf16:
                    {
                        auto vector = get_vector<bfloat16>();
                        rc = std::vector<T>(vector.begin(), vector.end());
                        break;
                    }
                    case element::Type_t::f16:
                    {
                        auto vector = get_vector<float16>();
                        rc = std::vector<T>(vector.begin(), vector.end());
                        break;
                    }
                    case element::Type_t::f32:
                    {
                        auto vector = get_vector<float>();
                        rc = std::vector<T>(vector.begin(), vector.end());
                        break;
                    }
                    case element::Type_t::f64:
                    {
                        auto vector = get_vector<double>();
                        rc = std::vector<T>(vector.begin(), vector.end());
                        break;
                    }
                    case element::Type_t::i8:
                    {
                        auto vector = get_vector<int8_t>();
                        rc = std::vector<T>(vector.begin(), vector.end());
                        break;
                    }
                    case element::Type_t::i16:
                    {
                        auto vector = get_vector<int16_t>();
                        rc = std::vector<T>(vector.begin(), vector.end());
                        break;
                    }
                    case element::Type_t::i32:
                    {
                        auto vector = get_vector<int32_t>();
                        rc = std::vector<T>(vector.begin(), vector.end());
                        break;
                    }
                    case element::Type_t::i64:
                    {
                        auto vector = get_vector<int64_t>();
                        rc = std::vector<T>(vector.begin(), vector.end());
                        break;
                    }
                    case element::Type_t::u8:
                    {
                        auto vector = get_vector<uint8_t>();
                        rc = std::vector<T>(vector.begin(), vector.end());
                        break;
                    }
                    case element::Type_t::u16:
                    {
                        auto vector = get_vector<uint16_t>();
                        rc = std::vector<T>(vector.begin(), vector.end());
                        break;
                    }
                    case element::Type_t::u32:
                    {
                        auto vector = get_vector<uint32_t>();
                        rc = std::vector<T>(vector.begin(), vector.end());
                        break;
                    }
                    case element::Type_t::u64:
                    {
                        auto vector = get_vector<uint64_t>();
                        rc = std::vector<T>(vector.begin(), vector.end());
                        break;
                    }
                    default: throw std::runtime_error("unsupported type");
                    }
                    return rc;
                }

                const void* get_data_ptr() const { return (m_data ? m_data->get_ptr() : nullptr); }
                template <typename T>
                const T* get_data_ptr() const
                {
                    return static_cast<const T*>(get_data_ptr());
                }

                template <element::Type_t ET>
                const typename element_type_traits<ET>::value_type* get_data_ptr() const
                {
                    NGRAPH_CHECK(ET == get_output_element_type(0),
                                 "get_data_ptr() called for incorrect element type.");
                    return static_cast<const typename element_type_traits<ET>::value_type*>(
                        get_data_ptr());
                }

                bool is_constant() const override { return true; }
                bool get_all_data_elements_bitwise_identical() const
                {
                    return m_all_elements_bitwise_identical;
                }
                std::string convert_value_to_string(size_t index) const;

            protected:
                /// \brief Allocate a buffer and return a pointer to it
                void* allocate_buffer();

                void* get_data_ptr_nc() { return (m_data ? m_data->get_ptr() : nullptr); }
                template <element::Type_t ET>
                typename element_type_traits<ET>::value_type* get_data_ptr_nc()
                {
                    NGRAPH_CHECK(ET == get_output_element_type(0),
                                 "get_data_ptr_nc() called for incorrect element type.");
                    return static_cast<typename element_type_traits<ET>::value_type*>(
                        get_data_ptr_nc());
                }

                Constant(const OutputVector& args)
                    : Op(args)
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
                                     const Shape& /* target_shape */,
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
                    case element::Type_t::u1: throw std::runtime_error("unsupported type");
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
                std::shared_ptr<runtime::AlignedBuffer> m_data;
                bool m_all_elements_bitwise_identical;
                bool are_all_data_elements_bitwise_identical() const;
            };

            /// \brief A scalar constant whose element type is the same as like.
            class NGRAPH_API ScalarConstantLike : public Constant
            {
            public:
                static constexpr NodeTypeInfo type_info{"ScalarConstantLike", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief A scalar constant whose element type is the same as like.
                ///
                /// Once the element type is known, the dependency on like will be removed and
                /// this node will be replaced with an equivalent constant.
                ///
                /// \param like A tensor that will supply the element type.
                /// \param value The value of the scalar.
                template <typename T>
                ScalarConstantLike(const Output<Node>& like, T value)
                    : Constant(OutputVector{like})
                    , m_value(static_cast<double>(value))
                {
                    constructor_validate_and_infer_types();
                }

                ScalarConstantLike() = default;

                std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
                std::shared_ptr<op::v0::Constant> as_constant() const;

            protected:
                void infer_element_type() override;

                double m_value;
            };
        }
    }
}
