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

#include <onnx-ml.pb.h>
#include <utility>
#include <vector>

#include "ngraph/op/constant.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        // Detecting automatically the underlying type used to store the information
        // for data type of values a tensor is holding. A bug was discovered in protobuf
        // which forced ONNX team to switch from `enum TensorProto_DataType` to `int32`
        // in order to workaround the bug. This line allows using both versions of ONNX
        // generated wrappers.
        using TensorProto_DataType = decltype(onnx::TensorProto{}.data_type());

        namespace error
        {
            namespace tensor
            {
                struct invalid_data_type : ngraph_error
                {
                    explicit invalid_data_type(TensorProto_DataType type)
                        : ngraph_error{"invalid data type: " +
                                       onnx::TensorProto_DataType_Name(
                                           static_cast<onnx::TensorProto_DataType>(type))}
                    {
                    }
                };

                struct unsupported_data_type : ngraph_error
                {
                    explicit unsupported_data_type(TensorProto_DataType type)
                        : ngraph_error{"unsupported data type: " +
                                       onnx::TensorProto_DataType_Name(
                                           static_cast<onnx::TensorProto_DataType>(type))}
                    {
                    }
                };

                struct unspecified_name : ngraph_error
                {
                    unspecified_name()
                        : ngraph_error{"tensor has no name specified"}
                    {
                    }
                };

                struct unspecified_data_type : ngraph_error
                {
                    unspecified_data_type()
                        : ngraph_error{"tensor has no data type specified"}
                    {
                    }
                };

                struct data_type_undefined : ngraph_error
                {
                    data_type_undefined()
                        : ngraph_error{"data type is not defined"}
                    {
                    }
                };

                struct segments_unsupported : ngraph_error
                {
                    segments_unsupported()
                        : ngraph_error{"loading segments not supported"}
                    {
                    }
                };

            } // namespace tensor

        } // namespace error

        namespace detail
        {
            namespace tensor
            {
                namespace
                {
                    namespace detail
                    {
                        template <typename T, typename Container>
                        inline std::vector<T> __get_data(const Container& container)
                        {
                            return {std::begin(container), std::end(container)};
                        }

                        template <typename T>
                        inline std::vector<T> __get_raw_data(const std::string& raw_data)
                        {
                            auto it = reinterpret_cast<const T*>(raw_data.data());
                            return {it, it + (raw_data.size() / sizeof(T))};
                        }
                    }
                }

                template <typename T>
                inline std::vector<T> get_data(const onnx::TensorProto& tensor)
                {
                    throw error::tensor::unsupported_data_type{tensor.data_type()};
                }

                template <>
                inline std::vector<double> get_data(const onnx::TensorProto& tensor)
                {
                    if (tensor.has_raw_data())
                    {
                        return detail::__get_raw_data<double>(tensor.raw_data());
                    }
                    if (tensor.data_type() == onnx::TensorProto_DataType_DOUBLE)
                    {
                        return detail::__get_data<double>(tensor.double_data());
                    }
                    if ((tensor.data_type() == onnx::TensorProto_DataType_FLOAT) or
                        (tensor.data_type() == onnx::TensorProto_DataType_FLOAT16))
                    {
                        return detail::__get_data<double>(tensor.float_data());
                    }
                    if (tensor.data_type() == onnx::TensorProto_DataType_INT32)
                    {
                        return detail::__get_data<double>(tensor.int32_data());
                    }
                    if (tensor.data_type() == onnx::TensorProto_DataType_INT64)
                    {
                        return detail::__get_data<double>(tensor.int64_data());
                    }
                    if (tensor.data_type() == onnx::TensorProto_DataType_UINT64)
                    {
                        return detail::__get_data<double>(tensor.uint64_data());
                    }
                    throw error::tensor::invalid_data_type{tensor.data_type()};
                }

                template <>
                inline std::vector<float> get_data(const onnx::TensorProto& tensor)
                {
                    if (tensor.has_raw_data())
                    {
                        return detail::__get_raw_data<float>(tensor.raw_data());
                    }
                    if ((tensor.data_type() == onnx::TensorProto_DataType_FLOAT) or
                        (tensor.data_type() == onnx::TensorProto_DataType_FLOAT16))
                    {
                        return detail::__get_data<float>(tensor.float_data());
                    }
                    if (tensor.data_type() == onnx::TensorProto_DataType_INT32)
                    {
                        return detail::__get_data<float>(tensor.int32_data());
                    }
                    if (tensor.data_type() == onnx::TensorProto_DataType_INT64)
                    {
                        return detail::__get_data<float>(tensor.int64_data());
                    }
                    if (tensor.data_type() == onnx::TensorProto_DataType_UINT64)
                    {
                        return detail::__get_data<float>(tensor.uint64_data());
                    }
                    throw error::tensor::invalid_data_type{tensor.data_type()};
                }

                template <>
                inline std::vector<int8_t> get_data(const onnx::TensorProto& tensor)
                {
                    if (tensor.has_raw_data())
                    {
                        return detail::__get_raw_data<int8_t>(tensor.raw_data());
                    }
                    if (tensor.data_type() == onnx::TensorProto_DataType_INT8)
                    {
                        return detail::__get_data<int8_t>(tensor.int32_data());
                    }
                    throw error::tensor::invalid_data_type{tensor.data_type()};
                }

                template <>
                inline std::vector<int16_t> get_data(const onnx::TensorProto& tensor)
                {
                    if (tensor.has_raw_data())
                    {
                        return detail::__get_raw_data<int16_t>(tensor.raw_data());
                    }
                    if (tensor.data_type() == onnx::TensorProto_DataType_INT16)
                    {
                        return detail::__get_data<int16_t>(tensor.int32_data());
                    }
                    throw error::tensor::invalid_data_type{tensor.data_type()};
                }

                template <>
                inline std::vector<int32_t> get_data(const onnx::TensorProto& tensor)
                {
                    if (tensor.has_raw_data())
                    {
                        return detail::__get_raw_data<int32_t>(tensor.raw_data());
                    }
                    if (tensor.data_type() == onnx::TensorProto_DataType_INT32)
                    {
                        return detail::__get_data<int32_t>(tensor.int32_data());
                    }
                    throw error::tensor::invalid_data_type{tensor.data_type()};
                }

                template <>
                inline std::vector<int64_t> get_data(const onnx::TensorProto& tensor)
                {
                    if (tensor.has_raw_data())
                    {
                        return detail::__get_raw_data<int64_t>(tensor.raw_data());
                    }
                    if (tensor.data_type() != onnx::TensorProto_DataType_INT64)
                    {
                        throw error::tensor::invalid_data_type{tensor.data_type()};
                    }
                    return detail::__get_data<int64_t>(tensor.int64_data());
                }

                template <>
                inline std::vector<uint8_t> get_data(const onnx::TensorProto& tensor)
                {
                    if (tensor.has_raw_data())
                    {
                        return detail::__get_raw_data<uint8_t>(tensor.raw_data());
                    }
                    if (tensor.data_type() == onnx::TensorProto_DataType_UINT8)
                    {
                        return detail::__get_data<uint8_t>(tensor.int32_data());
                    }
                    throw error::tensor::invalid_data_type{tensor.data_type()};
                }

                template <>
                inline std::vector<uint16_t> get_data(const onnx::TensorProto& tensor)
                {
                    if (tensor.has_raw_data())
                    {
                        return detail::__get_raw_data<uint16_t>(tensor.raw_data());
                    }
                    if (tensor.data_type() == onnx::TensorProto_DataType_UINT16)
                    {
                        return detail::__get_data<uint16_t>(tensor.int32_data());
                    }
                    throw error::tensor::invalid_data_type{tensor.data_type()};
                }

                template <>
                inline std::vector<uint32_t> get_data(const onnx::TensorProto& tensor)
                {
                    if (tensor.has_raw_data())
                    {
                        return detail::__get_raw_data<uint32_t>(tensor.raw_data());
                    }
                    if (tensor.data_type() == onnx::TensorProto_DataType_UINT32)
                    {
                        return detail::__get_data<uint32_t>(tensor.uint64_data());
                    }
                    throw error::tensor::invalid_data_type{tensor.data_type()};
                }

                template <>
                inline std::vector<uint64_t> get_data(const onnx::TensorProto& tensor)
                {
                    if (tensor.has_raw_data())
                    {
                        return detail::__get_raw_data<uint64_t>(tensor.raw_data());
                    }
                    if (tensor.data_type() != onnx::TensorProto_DataType_UINT64)
                    {
                        throw error::tensor::invalid_data_type{tensor.data_type()};
                    }
                    return detail::__get_data<uint64_t>(tensor.uint64_data());
                }
            }
        }

        class Tensor
        {
        public:
            enum class Type
            {
                undefined = onnx::TensorProto_DataType_UNDEFINED,
                float32 = onnx::TensorProto_DataType_FLOAT,
                uint8 = onnx::TensorProto_DataType_UINT8,
                int8 = onnx::TensorProto_DataType_INT8,
                uint16 = onnx::TensorProto_DataType_UINT16,
                int16 = onnx::TensorProto_DataType_INT16,
                int32 = onnx::TensorProto_DataType_INT32,
                int64 = onnx::TensorProto_DataType_INT64,
                string = onnx::TensorProto_DataType_STRING,
                boolean = onnx::TensorProto_DataType_BOOL,
                float16 = onnx::TensorProto_DataType_FLOAT16,
                float64 = onnx::TensorProto_DataType_DOUBLE,
                uint32 = onnx::TensorProto_DataType_UINT32,
                uint64 = onnx::TensorProto_DataType_UINT64,
                complex64 = onnx::TensorProto_DataType_COMPLEX64,
                complex128 = onnx::TensorProto_DataType_COMPLEX128
            };

            Tensor() = delete;
            explicit Tensor(const onnx::TensorProto& tensor)
                : m_tensor_proto{&tensor}
                , m_shape{std::begin(tensor.dims()), std::end(tensor.dims())}
            {
            }

            Tensor(const Tensor&) = default;
            Tensor(Tensor&&) = default;

            Tensor& operator=(const Tensor&) = delete;
            Tensor& operator=(Tensor&&) = delete;

            const Shape& get_shape() const { return m_shape; }
            template <typename T>
            std::vector<T> get_data() const
            {
                if (m_tensor_proto->has_segment())
                {
                    throw error::tensor::segments_unsupported{};
                }
                return detail::tensor::get_data<T>(*m_tensor_proto);
            }

            const std::string& get_name() const
            {
                if (!m_tensor_proto->has_name())
                {
                    throw error::tensor::unspecified_name{};
                }
                return m_tensor_proto->name();
            }

            Type get_type() const
            {
                if (!m_tensor_proto->has_data_type())
                {
                    throw error::tensor::unspecified_data_type{};
                }
                return static_cast<Type>(m_tensor_proto->data_type());
            }

            const element::Type& get_ng_type() const
            {
                if (!m_tensor_proto->has_data_type())
                {
                    throw error::tensor::unspecified_data_type{};
                }
                switch (m_tensor_proto->data_type())
                {
                case onnx::TensorProto_DataType::TensorProto_DataType_BOOL: return element::boolean;
                case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT:
                case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT16: return element::f32;
                case onnx::TensorProto_DataType::TensorProto_DataType_DOUBLE: return element::f64;
                case onnx::TensorProto_DataType::TensorProto_DataType_INT8: return element::i8;
                case onnx::TensorProto_DataType::TensorProto_DataType_INT16: return element::i16;
                case onnx::TensorProto_DataType::TensorProto_DataType_INT32: return element::i32;
                case onnx::TensorProto_DataType::TensorProto_DataType_INT64: return element::i64;
                case onnx::TensorProto_DataType::TensorProto_DataType_UINT8: return element::u8;
                case onnx::TensorProto_DataType::TensorProto_DataType_UINT16: return element::u16;
                case onnx::TensorProto_DataType::TensorProto_DataType_UINT32: return element::u32;
                case onnx::TensorProto_DataType::TensorProto_DataType_UINT64: return element::u64;
                case onnx::TensorProto_DataType::TensorProto_DataType_UNDEFINED:
                    throw error::tensor::data_type_undefined{};
                default: throw error::tensor::unsupported_data_type{m_tensor_proto->data_type()};
                }
            }

            operator TensorProto_DataType() const { return m_tensor_proto->data_type(); }
            std::shared_ptr<ngraph::op::Constant> get_ng_constant() const
            {
                switch (m_tensor_proto->data_type())
                {
                case onnx::TensorProto_DataType::TensorProto_DataType_BOOL:
                    return make_ng_constant<bool>(element::boolean);
                case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT:
                case onnx::TensorProto_DataType::TensorProto_DataType_FLOAT16:
                    return make_ng_constant<float>(element::f32);
                case onnx::TensorProto_DataType::TensorProto_DataType_DOUBLE:
                    return make_ng_constant<double>(element::f64);
                case onnx::TensorProto_DataType::TensorProto_DataType_INT8:
                    return make_ng_constant<int8_t>(element::i8);
                case onnx::TensorProto_DataType::TensorProto_DataType_INT16:
                    return make_ng_constant<int16_t>(element::i16);
                case onnx::TensorProto_DataType::TensorProto_DataType_INT32:
                    return make_ng_constant<int32_t>(element::i32);
                case onnx::TensorProto_DataType::TensorProto_DataType_INT64:
                    return make_ng_constant<int64_t>(element::i64);
                case onnx::TensorProto_DataType::TensorProto_DataType_UINT8:
                    return make_ng_constant<uint8_t>(element::u8);
                case onnx::TensorProto_DataType::TensorProto_DataType_UINT16:
                    return make_ng_constant<uint16_t>(element::u16);
                case onnx::TensorProto_DataType::TensorProto_DataType_UINT32:
                    return make_ng_constant<uint32_t>(element::u32);
                case onnx::TensorProto_DataType::TensorProto_DataType_UINT64:
                    return make_ng_constant<uint64_t>(element::u64);
                default: throw error::tensor::unsupported_data_type{m_tensor_proto->data_type()};
                }
            }

        private:
            template <typename T>
            std::shared_ptr<ngraph::op::Constant> make_ng_constant(const element::Type& type) const
            {
                return std::make_shared<ngraph::op::Constant>(type, m_shape, get_data<T>());
            }

            const onnx::TensorProto* m_tensor_proto;
            Shape m_shape;
        };

        inline std::ostream& operator<<(std::ostream& outs, const Tensor& tensor)
        {
            return (outs << "<Tensor: " << tensor.get_name() << ">");
        }

    } // namespace onnx_import

} // namespace ngraph
