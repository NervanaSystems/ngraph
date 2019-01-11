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

#include <memory>

#include <cuda_runtime.h>

#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/runtime/nvidiagpu/backend.hpp"
#include "ngraph/runtime/nvidiagpu/cuda_error_check.hpp"
#include "ngraph/runtime/nvidiagpu/tensor.hpp"
#include "ngraph/runtime/nvidiagpu/util.hpp"

using namespace ngraph;
using namespace std;

runtime::nvidiagpu::Tensor::Tensor(const ngraph::element::Type& element_type,
                                   const ngraph::Shape& shape,
                                   void* memory_pointer,
                                   const ngraph::runtime::Backend* backend)
    : runtime::Tensor(std::make_shared<ngraph::descriptor::Tensor>(element_type, shape, "external"),
                      backend)
    , m_custom_memory(false)
{
    m_descriptor->set_tensor_layout(
        std::make_shared<ngraph::descriptor::layout::DenseTensorLayout>(*m_descriptor));

    m_buffer_size = shape_size(shape) * element_type.size();
    if (memory_pointer != nullptr)
    {
        m_allocated_buffer_pool = memory_pointer;
        m_custom_memory = true;
    }
    else if (m_buffer_size > 0)
    {
        m_allocated_buffer_pool = runtime::nvidiagpu::create_nvidiagpu_buffer(m_buffer_size);
    }
}

runtime::nvidiagpu::Tensor::Tensor(const ngraph::element::Type& element_type,
                                   const ngraph::Shape& shape,
                                   const ngraph::runtime::Backend* backend)
    : nvidiagpu::Tensor(element_type, shape, nullptr, backend)
{
}

runtime::nvidiagpu::Tensor::~Tensor()
{
    if (!m_custom_memory && (m_allocated_buffer_pool != nullptr))
    {
        runtime::nvidiagpu::free_nvidiagpu_buffer(m_allocated_buffer_pool);
    }
}

void runtime::nvidiagpu::Tensor::write(const void* source, size_t tensor_offset, size_t n)
{
    CUDA_RT_SAFE_CALL(cudaMemcpy(m_allocated_buffer_pool, source, n, cudaMemcpyHostToDevice));
}

void runtime::nvidiagpu::Tensor::read(void* target, size_t tensor_offset, size_t n) const
{
    CUDA_RT_SAFE_CALL(cudaMemcpy(target, m_allocated_buffer_pool, n, cudaMemcpyDeviceToHost));
}
