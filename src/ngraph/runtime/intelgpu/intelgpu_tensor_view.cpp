/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <memory>

#include <CPP/data.hpp>

#include "ngraph/descriptor/layout/dense_tensor_view_layout.hpp"
#include "ngraph/descriptor/primary_tensor_view.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_tensor_view.hpp"
#include "ngraph/shape.hpp"

using namespace ngraph;
using namespace std;

runtime::intelgpu::IntelGPUTensorView::IntelGPUTensorView(const ngraph::element::Type& element_type,
                                                          const Shape& shape,
                                                          const cldnn::engine& backend_engine,
                                                          void* memory_pointer)
    : runtime::TensorView(std::make_shared<ngraph::descriptor::PrimaryTensorView>(
          std::make_shared<ngraph::TensorViewType>(element_type, shape), "external"))
{
    m_descriptor->set_tensor_view_layout(
        std::make_shared<ngraph::descriptor::layout::DenseTensorViewLayout>(*m_descriptor));

    size_t mem_size = shape_size(shape);
    cldnn::data_types data_type = get_cldnn_type(element_type);
    cldnn::tensor tensor(1, mem_size, 1, 1);
    cldnn::format::type format = cldnn::format::yxfb;

    cldnn::layout layout(data_type, format, tensor);

    if (nullptr != memory_pointer)
    {
        ocl_memory = make_shared<cldnn::memory>(
            cldnn::memory::attach<void>(layout, memory_pointer, layout.get_linear_size()));
    }
    else
    {
        ocl_memory = make_shared<cldnn::memory>(cldnn::memory::allocate(backend_engine, layout));
    }
}

void runtime::intelgpu::IntelGPUTensorView::write(const void* source,
                                                  size_t tensor_offset,
                                                  size_t n)
{
    if (tensor_offset + n > ocl_memory->size())
    {
        throw out_of_range("write access past end of tensor");
    }

    auto ptr = ocl_memory->pointer<char>();
    char* target = ptr.data();
    memcpy(&target[tensor_offset], source, n);
}

void runtime::intelgpu::IntelGPUTensorView::read(void* target, size_t tensor_offset, size_t n) const
{
    if (tensor_offset + n > ocl_memory->size())
    {
        throw out_of_range("read access past end of tensor");
    }

    const auto ptr = ocl_memory->pointer<char>();
    const char* source = ptr.data();
    memcpy(target, &source[tensor_offset], n);
}

cldnn::data_types runtime::intelgpu::IntelGPUTensorView::get_cldnn_type(
    const ngraph::element::Type& element_type) const
{
    if (element_type == ngraph::element::i8)
    {
        return cldnn::data_types::i8;
    }
    else if (element_type == ngraph::element::u8)
    {
        return cldnn::data_types::u8;
    }
    else if (element_type == ngraph::element::f32)
    {
        return cldnn::data_types::f32;
    }
    else
    {
        ostringstream os;
        os << "IntelGPUTensorView::get_cldnn_type: Unknown type " << element_type;
        throw std::invalid_argument(os.str());
    }
}
