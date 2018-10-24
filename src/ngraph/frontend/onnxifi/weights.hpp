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

#include <onnx.hpp>
#include <onnxifi.h>

#include "span.hpp"

namespace ngraph
{
    namespace onnxifi
    {
        class Weight
        {
        public:
            Weight(const Weight&) = default;
            Weight& operator=(const Weight&) = default;

            Weight(Weight&&) = default;
            Weight& operator=(Weight&&) = default;

            Weight() = delete;
            explicit Weight(const ::onnxTensorDescriptorV1& weight)
                : m_name{weight.name},
                  m_type{get_type(weight.dataType)}
            {
                if (weight.tag != ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1)
                {
                    throw status::unsupported_tag{};
                }
                if (weight.name == nullptr)
                {
                    throw status::invalid_name{};
                }
                switch (weight.memoryType)
                {
                    case ONNXIFI_MEMORY_TYPE_CPU: break;
                    case ONNXIFI_MEMORY_TYPE_CUDA_BUFFER:
                    case ONNXIFI_MEMORY_TYPE_OPENCL_BUFFER:
                    case ONNXIFI_MEMORY_TYPE_OPENGLES_TEXTURE_2D:
                    case ONNXIFI_MEMORY_TYPE_D3D_RESOURCE: throw status::invalid_memory_type{};
                    default: throw status::unsupported_memory_type{};
                }
                if ((weight.dimensions != 0) && (weight.shape == nullptr))
                {
                    throw status::null_pointer{};
                }
                if ((weight.shape != nullptr) && (weight.dimensions == 0))
                {
                    throw status::invalid_size{};
                }
                if (weight.shape == nullptr)
                {
                    m_shape = {1};
                }
                else
                {
                    Span<uint64_t> shape{weight.shape, weight.dimensions};
                    for (const auto& value : shape)
                    {
                        if (value == 0)
                        {
                            throw status::invalid_shape{};
                        }
                        m_shape.push_back(value);
                        m_size *= value;
                    }
                }
                if (weight.buffer == 0)
                {
                    throw status::invalid_memory_location{};
                }
                const char* buffer{reinterpret_cast<const char*>(weight.buffer)};
                m_buffer.assign(buffer, buffer + (m_size * get_type_size(weight.dataType)));
            }

            onnx_import::Weight get() const
            {
                return {m_type, m_shape.size(), m_shape.data(), m_buffer.data()};
            }

            const std::string& name() const
            {
                return m_name;
            }

        private:
            std::string m_name{};
            Shape m_shape;
            std::size_t m_size{1};
            onnx_import::Weight::Type m_type;
            std::vector<char> m_buffer;

            onnx_import::Weight::Type get_type(::onnxEnum type)
            {
                switch (type)
                {
                case ONNXIFI_DATATYPE_FLOAT16:
                case ONNXIFI_DATATYPE_FLOAT32:
                    return onnx_import::Weight::Type::f32;
                case ONNXIFI_DATATYPE_FLOAT64:
                    return onnx_import::Weight::Type::f64;
                case ONNXIFI_DATATYPE_INT8:
                    return onnx_import::Weight::Type::i8;
                case ONNXIFI_DATATYPE_INT16:
                    return onnx_import::Weight::Type::i16;
                case ONNXIFI_DATATYPE_INT32:
                    return onnx_import::Weight::Type::i32;
                case ONNXIFI_DATATYPE_INT64:
                    return onnx_import::Weight::Type::i64;
                case ONNXIFI_DATATYPE_UINT8:
                    return onnx_import::Weight::Type::u8;
                case ONNXIFI_DATATYPE_UINT16:
                    return onnx_import::Weight::Type::u16;
                case ONNXIFI_DATATYPE_UINT32:
                    return onnx_import::Weight::Type::u32;
                case ONNXIFI_DATATYPE_UINT64:
                    return onnx_import::Weight::Type::u64;
                case ONNXIFI_DATATYPE_COMPLEX64:
                case ONNXIFI_DATATYPE_COMPLEX128:
                    throw status::invalid_datatype{};
                default:
                    throw status::unsupported_datatype{};
                }
            }

            std::size_t get_type_size(::onnxEnum type)
            {
                switch (type)
                {
                case ONNXIFI_DATATYPE_FLOAT16:
                case ONNXIFI_DATATYPE_FLOAT32:
                    return sizeof(float);
                case ONNXIFI_DATATYPE_FLOAT64:
                    return sizeof(double);
                case ONNXIFI_DATATYPE_INT8:
                    return sizeof(int8_t);
                case ONNXIFI_DATATYPE_INT16:
                    return sizeof(int16_t);
                case ONNXIFI_DATATYPE_INT32:
                    return sizeof(int32_t);
                case ONNXIFI_DATATYPE_INT64:
                    return sizeof(int64_t);
                case ONNXIFI_DATATYPE_UINT8:
                    return sizeof(uint8_t);
                case ONNXIFI_DATATYPE_UINT16:
                    return sizeof(uint16_t);
                case ONNXIFI_DATATYPE_UINT32:
                    return sizeof(uint32_t);
                case ONNXIFI_DATATYPE_UINT64:
                    return sizeof(uint64_t);
                case ONNXIFI_DATATYPE_COMPLEX64:
                case ONNXIFI_DATATYPE_COMPLEX128:
                    throw status::invalid_datatype{};
                default:
                    throw status::unsupported_datatype{};
                }
            }
        };
    }
}
