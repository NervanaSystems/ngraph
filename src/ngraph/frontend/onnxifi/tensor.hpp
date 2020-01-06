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

#include <memory>
#include <onnx/onnxifi.h>

#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/tensor.hpp"

namespace ngraph
{
    namespace onnxifi
    {
        /// \brief Wrapper for onnxTensorDescriptorV1 class
        class Tensor
        {
        public:
            Tensor(const Tensor&) = default;
            Tensor& operator=(const Tensor&) = default;

            Tensor(Tensor&&) = default;
            Tensor& operator=(Tensor&&) = default;

            Tensor() = delete;
            virtual ~Tensor() = default;

            explicit Tensor(const ::onnxTensorDescriptorV1& tensor);

            /// \brief Convert to ngraph::runtime::Tensor
            /// This function method converts ONNXIFI tensor to nGraph tensor.
            /// \param backend     the backend to use for nGraph tensor creation.
            /// \returns Shared pointer to nGraph tensor.
            std::shared_ptr<runtime::Tensor> to_ng(runtime::Backend& backend) const;

            /// \brief Copies data from ngraph::runtime::Tensor
            /// This function method writes the content of nGraph tensor.
            /// \param tensor     nGraph tensor to copy from.
            void from_ng(const runtime::Tensor& tensor);

            const void* data() const { return reinterpret_cast<const void*>(m_tensor->buffer); }
            std::size_t size() const { return m_size; }
            const Shape& get_shape() const { return m_shape; }
            const char* get_name() const { return m_tensor->name; }
        protected:
            const ::onnxTensorDescriptorV1* m_tensor;
            Shape m_shape;
            std::size_t m_size{1};
        };

    } // namespace onnxifi

} // namespace ngraph
