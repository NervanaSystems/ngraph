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

#include "tensor.hpp"
#include "exceptions.hpp"
#include "span.hpp"

namespace ngraph
{
    namespace onnxifi
    {
        Tensor::Tensor(const ::onnxTensorDescriptorV1& tensor)
            : m_tensor{&tensor}
        {
            if (tensor.tag != ONNXIFI_TAG_TENSOR_DESCRIPTOR_V1)
            {
                throw status::unsupported_tag{};
            }
            if (tensor.name == nullptr)
            {
                throw status::invalid_name{};
            }
            switch (tensor.dataType)
            {
            case ONNXIFI_DATATYPE_FLOAT16:
            case ONNXIFI_DATATYPE_FLOAT32:
            case ONNXIFI_DATATYPE_FLOAT64:
            case ONNXIFI_DATATYPE_INT8:
            case ONNXIFI_DATATYPE_INT16:
            case ONNXIFI_DATATYPE_INT32:
            case ONNXIFI_DATATYPE_INT64:
            case ONNXIFI_DATATYPE_UINT8:
            case ONNXIFI_DATATYPE_UINT16:
            case ONNXIFI_DATATYPE_UINT32:
            case ONNXIFI_DATATYPE_UINT64: break;
            case ONNXIFI_DATATYPE_COMPLEX64:
            case ONNXIFI_DATATYPE_COMPLEX128: throw status::invalid_datatype{};
            default: throw status::unsupported_datatype{};
            }
            switch (tensor.memoryType)
            {
            case ONNXIFI_MEMORY_TYPE_CPU: break;
            case ONNXIFI_MEMORY_TYPE_CUDA_BUFFER:
            case ONNXIFI_MEMORY_TYPE_OPENCL_BUFFER:
            case ONNXIFI_MEMORY_TYPE_OPENGLES_TEXTURE_2D:
            case ONNXIFI_MEMORY_TYPE_D3D_RESOURCE: throw status::invalid_memory_type{};
            default: throw status::unsupported_memory_type{};
            }
            if ((tensor.dimensions != 0) && (tensor.shape == nullptr))
            {
                throw status::null_pointer{};
            }
            if ((tensor.shape != nullptr) && (tensor.dimensions == 0))
            {
                throw status::invalid_size{};
            }
            if (tensor.shape == nullptr)
            {
                m_shape = {1};
            }
            else
            {
                Span<uint64_t> shape{tensor.shape, tensor.dimensions};
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
            if (tensor.buffer == 0)
            {
                throw status::invalid_memory_location{};
            }
        }

        std::shared_ptr<runtime::Tensor> Tensor::to_ng(runtime::Backend& backend) const
        {
            std::shared_ptr<runtime::Tensor> tensor;
            switch (m_tensor->dataType)
            {
            case ONNXIFI_DATATYPE_FLOAT16:
            case ONNXIFI_DATATYPE_FLOAT32:
                tensor = backend.create_tensor(element::f32, m_shape);
                tensor->write(data(), sizeof(float) * size());
                break;
            case ONNXIFI_DATATYPE_FLOAT64:
                tensor = backend.create_tensor(element::f64, m_shape);
                tensor->write(data(), sizeof(double) * size());
                break;
            case ONNXIFI_DATATYPE_INT8:
                tensor = backend.create_tensor(element::i8, m_shape);
                tensor->write(data(), sizeof(int8_t) * size());
                break;
            case ONNXIFI_DATATYPE_INT16:
                tensor = backend.create_tensor(element::i16, m_shape);
                tensor->write(data(), sizeof(int16_t) * size());
                break;
            case ONNXIFI_DATATYPE_INT32:
                tensor = backend.create_tensor(element::i32, m_shape);
                tensor->write(data(), sizeof(int32_t) * size());
                break;
            case ONNXIFI_DATATYPE_INT64:
                tensor = backend.create_tensor(element::i64, m_shape);
                tensor->write(data(), sizeof(int64_t) * size());
                break;
            case ONNXIFI_DATATYPE_UINT8:
                tensor = backend.create_tensor(element::u8, m_shape);
                tensor->write(data(), sizeof(uint8_t) * size());
                break;
            case ONNXIFI_DATATYPE_UINT16:
                tensor = backend.create_tensor(element::u16, m_shape);
                tensor->write(data(), sizeof(uint16_t) * size());
                break;
            case ONNXIFI_DATATYPE_UINT32:
                tensor = backend.create_tensor(element::u32, m_shape);
                tensor->write(data(), sizeof(uint32_t) * size());
                break;
            case ONNXIFI_DATATYPE_UINT64:
                tensor = backend.create_tensor(element::u64, m_shape);
                tensor->write(data(), sizeof(uint64_t) * size());
                break;
            default: throw status::unsupported_datatype{};
            }
            return tensor;
        }

        void Tensor::from_ng(const runtime::Tensor& tensor)
        {
            std::size_t readSize{tensor.get_element_count()};
            switch (m_tensor->dataType)
            {
            case ONNXIFI_DATATYPE_FLOAT16:
            case ONNXIFI_DATATYPE_FLOAT32: readSize *= sizeof(float); break;
            case ONNXIFI_DATATYPE_FLOAT64: readSize *= sizeof(double); break;
            case ONNXIFI_DATATYPE_INT8: readSize *= sizeof(int8_t); break;
            case ONNXIFI_DATATYPE_INT16: readSize *= sizeof(int16_t); break;
            case ONNXIFI_DATATYPE_INT32: readSize *= sizeof(int32_t); break;
            case ONNXIFI_DATATYPE_INT64: readSize *= sizeof(int64_t); break;
            case ONNXIFI_DATATYPE_UINT8: readSize *= sizeof(uint8_t); break;
            case ONNXIFI_DATATYPE_UINT16: readSize *= sizeof(uint16_t); break;
            case ONNXIFI_DATATYPE_UINT32: readSize *= sizeof(uint32_t); break;
            case ONNXIFI_DATATYPE_UINT64: readSize *= sizeof(uint64_t); break;
            default: break;
            }
            tensor.read(reinterpret_cast<void*>(m_tensor->buffer), readSize);
        }

    } // namespace onnxifi

} // namespace ngraph
