// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <memory>

#include <cuda.h>

#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/descriptor/primary_tensor_view.hpp"
#include "ngraph/runtime/gpu/gpu_backend.hpp"
#include "ngraph/runtime/gpu/gpu_tensor_view.hpp"

using namespace ngraph;
using namespace std;

runtime::gpu::GPU_TensorView::GPU_TensorView(const ngraph::element::Type& element_type,
                                             const Shape& shape)
    : runtime::TensorView(std::make_shared<ngraph::descriptor::PrimaryTensorView>(
          std::make_shared<ngraph::TensorViewType>(element_type, shape),
          "external",
          true,
          true,
          false))
{
    // Need to check type and have host/device tensors
    m_descriptor->set_tensor_view_layout(
        std::make_shared<ngraph::descriptor::layout::DenseTensorViewLayout>(*m_descriptor));

    m_buffer_size = m_descriptor->get_tensor_view_layout()->get_size() * element_type.size();

    // cuMemAlloc(&dev_buffer, m_buffer_size);
}

runtime::gpu::GPU_TensorView::~GPU_TensorView()
{
    // cuMemFree(dev_buffer);
}
void runtime::gpu::GPU_TensorView::write(const void* source, size_t tensor_offset, size_t n)
{
    // cuMemcpyHtoD(dev_buffer, source, n);
}

void runtime::gpu::GPU_TensorView::read(void* target, size_t tensor_offset, size_t n) const
{
    // cuMemcpyDtoH(target, dev_buffer, n);
}
