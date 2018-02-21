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

#include <cstring>
#include <memory>

#include "cpu_tensor_view.hpp"
#include "ngraph/descriptor/layout/tensor_view_layout.hpp"
#include "ngraph/descriptor/primary_tensor_view.hpp"
#include "ngraph/except.hpp"
#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
#include "ngraph/shape.hpp"

using namespace ngraph;
using namespace std;

// TODO(jmenon): Refactor all the alignment specifications into
// a single place and allow lower or no alignment when possible

const size_t runtime::cpu::CPUTensorView::s_buffer_alignment = 64;

runtime::cpu::CPUTensorView::CPUTensorView(const ngraph::element::Type& element_type,
                                           const Shape& shape,
                                           const string& name)
    : runtime::TensorView(std::make_shared<ngraph::descriptor::PrimaryTensorView>(
          std::make_shared<ngraph::TensorViewType>(element_type, shape), name, true, true, false))
    , m_buffer(nullptr)
    , m_aligned_buffer(nullptr)
{
    // TODO(jmenon): A fallback layout should not be needed but is required
    // because of how some unit test functionality is written (ex. 'backprop_derivative')
    // This needs to be removed
    m_descriptor->set_tensor_view_layout(std::make_shared<runtime::cpu::LayoutDescriptor>(
        *m_descriptor, runtime::cpu::LayoutDescriptor::create_native_axis_order(shape.size())));

    m_buffer_size = shape_size(shape) * element_type.size();
    if (m_buffer_size)
    {
        size_t allocation_size = m_buffer_size + s_buffer_alignment;
        auto ptr = malloc(allocation_size);
        if (!ptr)
        {
            throw ngraph_error("Error allocating CPU Tensor View memory");
        }
        m_buffer = static_cast<char*>(ptr);

// GCC major versions below 5 do not implement C++11 std::align
#if !defined(__GNUC__) || __GNUC__ >= 5
        std::align(s_buffer_alignment, m_buffer_size, ptr, allocation_size);
#else
        ptr = static_cast<char*>(ptr) + (s_buffer_alignment - 1);
        ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(ptr) &
                                      ~(uintptr_t(s_buffer_alignment - 1)));
#endif

        m_aligned_buffer = static_cast<char*>(ptr);
    }
}

runtime::cpu::CPUTensorView::~CPUTensorView()
{
    free(m_buffer);
}

char* runtime::cpu::CPUTensorView::get_data_ptr()
{
    return m_aligned_buffer;
}

const char* runtime::cpu::CPUTensorView::get_data_ptr() const
{
    return m_aligned_buffer;
}

void runtime::cpu::CPUTensorView::write(const void* source, size_t tensor_offset, size_t n)
{
    if (tensor_offset + n > m_buffer_size)
    {
        throw out_of_range("write access past end of tensor");
    }
    char* target = get_data_ptr();
    memcpy(&target[tensor_offset], source, n);
}

void runtime::cpu::CPUTensorView::read(void* target, size_t tensor_offset, size_t n) const
{
    if (tensor_offset + n > m_buffer_size)
    {
        throw out_of_range("read access past end of tensor");
    }
    const char* source = get_data_ptr();
    memcpy(target, &source[tensor_offset], n);
}

size_t runtime::cpu::CPUTensorView::get_size() const
{
    return get_element_count();
}
