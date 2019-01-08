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
#include "ngraph/runtime/nvgpu/cuda_error_check.hpp"
#include "ngraph/runtime/nvgpu/nvgpu_backend.hpp"
#include "ngraph/runtime/nvgpu/nvgpu_tensor.hpp"
#include "ngraph/runtime/nvgpu/nvgpu_util.hpp"

using namespace ngraph;
using namespace std;

runtime::nvgpu::NVTensor::NVTensor(const ngraph::element::Type& element_type,
                                   const Shape& shape,
                                   void* memory_pointer,
                                   const Backend* backend)
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
        m_allocated_buffer_pool = runtime::nvgpu::create_nvgpu_buffer(m_buffer_size);
    }
}

runtime::nvgpu::NVTensor::NVTensor(const ngraph::element::Type& element_type,
                                   const Shape& shape,
                                   const Backend* backend)
    : NVTensor(element_type, shape, nullptr, backend)
{
}

runtime::nvgpu::NVTensor::~NVTensor()
{
    if (!m_custom_memory && (m_allocated_buffer_pool != nullptr))
    {
        runtime::nvgpu::free_nvgpu_buffer(m_allocated_buffer_pool);
    }
}

void runtime::nvgpu::NVTensor::write(const void* source, size_t tensor_offset, size_t n)
{
    CUDA_RT_SAFE_CALL(cudaMemcpy(m_allocated_buffer_pool, source, n, cudaMemcpyHostToDevice));
}

void runtime::nvgpu::NVTensor::read(void* target, size_t tensor_offset, size_t n) const
{
    CUDA_RT_SAFE_CALL(cudaMemcpy(target, m_allocated_buffer_pool, n, cudaMemcpyDeviceToHost));
}
