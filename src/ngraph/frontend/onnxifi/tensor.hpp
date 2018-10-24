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

#include <memory>

#include <onnxifi.h>

#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/tensor.hpp"

#include "exceptions.hpp"
#include "span.hpp"

namespace ngraph
{
    namespace onnxifi
    {
        /// \brief Wrapper to onnxTensorDescriptorV1 class
        class Tensor
        {
        public:
            Tensor(const Tensor&) = default;
            Tensor& operator=(const Tensor&) = default;

            Tensor(Tensor&&) noexcept = default;
            Tensor& operator=(Tensor&&) noexcept = default;

            Tensor() = delete;

            explicit Tensor(const ::onnxTensorDescriptorV1& tensor)
                : m_tensor{tensor}
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

            virtual std::shared_ptr<runtime::Tensor> to_ng(runtime::Backend& backend) const = 0;

            const void* data() const { return reinterpret_cast<const void*>(m_tensor.buffer); }
            std::size_t size() const { return m_size; }
            const Shape& get_shape() const { return m_shape; }
        protected:
            const ::onnxTensorDescriptorV1& m_tensor;
            Shape m_shape;
            std::size_t m_size{1};
        };

        class InputTensor final : public Tensor
        {
        public:
            using Tensor::Tensor;
            std::shared_ptr<runtime::Tensor> to_ng(runtime::Backend& backend) const final;
        };

        class OutputTensor final : public Tensor
        {
        public:
            using Tensor::Tensor;
            std::shared_ptr<runtime::Tensor> to_ng(runtime::Backend& backend) const final;
            void from_ng(const runtime::Tensor& tensor);
        };
    }
}
