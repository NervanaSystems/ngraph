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

#include "tensor.hpp"

namespace ngraph
{
    namespace onnxifi
    {
        std::shared_ptr<runtime::Tensor> InputTensor::to_ng(runtime::Backend& backend) const
        {
            std::shared_ptr<runtime::Tensor> tensor;
            switch (m_tensor->dataType)
            {
            case ONNXIFI_DATATYPE_FLOAT16:
            case ONNXIFI_DATATYPE_FLOAT32:
                tensor = backend.create_tensor(element::f32, m_shape);
                tensor->write(data(), 0, sizeof(float) * size());
                break;
            case ONNXIFI_DATATYPE_FLOAT64:
                tensor = backend.create_tensor(element::f64, m_shape);
                tensor->write(data(), 0, sizeof(double) * size());
                break;
            case ONNXIFI_DATATYPE_INT8:
                tensor = backend.create_tensor(element::i8, m_shape);
                tensor->write(data(), 0, sizeof(int8_t) * size());
                break;
            case ONNXIFI_DATATYPE_INT16:
                tensor = backend.create_tensor(element::i16, m_shape);
                tensor->write(data(), 0, sizeof(int16_t) * size());
                break;
            case ONNXIFI_DATATYPE_INT32:
                tensor = backend.create_tensor(element::i32, m_shape);
                tensor->write(data(), 0, sizeof(int32_t) * size());
                break;
            case ONNXIFI_DATATYPE_INT64:
                tensor = backend.create_tensor(element::i64, m_shape);
                tensor->write(data(), 0, sizeof(int64_t) * size());
                break;
            case ONNXIFI_DATATYPE_UINT8:
                tensor = backend.create_tensor(element::u8, m_shape);
                tensor->write(data(), 0, sizeof(uint8_t) * size());
                break;
            case ONNXIFI_DATATYPE_UINT16:
                tensor = backend.create_tensor(element::u16, m_shape);
                tensor->write(data(), 0, sizeof(uint16_t) * size());
                break;
            case ONNXIFI_DATATYPE_UINT32:
                tensor = backend.create_tensor(element::u32, m_shape);
                tensor->write(data(), 0, sizeof(uint32_t) * size());
                break;
            case ONNXIFI_DATATYPE_UINT64:
                tensor = backend.create_tensor(element::u64, m_shape);
                tensor->write(data(), 0, sizeof(uint64_t) * size());
                break;
            default: throw status::unsupported_datatype{};
            }
            return tensor;
        }

        std::shared_ptr<runtime::Tensor> OutputTensor::to_ng(runtime::Backend& backend) const
        {
            switch (m_tensor->dataType)
            {
            case ONNXIFI_DATATYPE_FLOAT16:
            case ONNXIFI_DATATYPE_FLOAT32: return backend.create_tensor(element::f32, m_shape);
            case ONNXIFI_DATATYPE_FLOAT64: return backend.create_tensor(element::f64, m_shape);
            case ONNXIFI_DATATYPE_INT8: return backend.create_tensor(element::i8, m_shape);
            case ONNXIFI_DATATYPE_INT16: return backend.create_tensor(element::i16, m_shape);
            case ONNXIFI_DATATYPE_INT32: return backend.create_tensor(element::i32, m_shape);
            case ONNXIFI_DATATYPE_INT64: return backend.create_tensor(element::i64, m_shape);
            case ONNXIFI_DATATYPE_UINT8: return backend.create_tensor(element::u8, m_shape);
            case ONNXIFI_DATATYPE_UINT16: return backend.create_tensor(element::u16, m_shape);
            case ONNXIFI_DATATYPE_UINT32: return backend.create_tensor(element::u32, m_shape);
            case ONNXIFI_DATATYPE_UINT64: return backend.create_tensor(element::u64, m_shape);
            default: throw status::unsupported_datatype{};
            }
        }

        void OutputTensor::from_ng(const runtime::Tensor& tensor)
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
            tensor.read(reinterpret_cast<void*>(m_tensor->buffer), 0, readSize);
        }

    } // namespace onnxifi

} // namespace ngraph
