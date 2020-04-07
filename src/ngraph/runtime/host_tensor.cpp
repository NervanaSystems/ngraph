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

#include <cstring>
#include <memory>

#include "ngraph/chrome_trace.hpp"
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

static const size_t alignment = 64;

runtime::HostTensor::HostTensor(const ngraph::element::Type& element_type,
                                const Shape& shape,
                                void* memory_pointer,
                                const string& name)
    : runtime::Tensor(std::make_shared<ngraph::descriptor::Tensor>(element_type, shape, name))
    , m_allocated_buffer_pool(nullptr)
    , m_aligned_buffer_pool(nullptr)

{
    m_descriptor->set_tensor_layout(
        std::make_shared<ngraph::descriptor::layout::DenseTensorLayout>(*m_descriptor));

    m_buffer_size = m_descriptor->get_tensor_layout()->get_size() * element_type.size();

    if (memory_pointer != nullptr)
    {
        m_aligned_buffer_pool = static_cast<char*>(memory_pointer);
    }
    else if (m_buffer_size > 0)
    {
        size_t allocation_size = m_buffer_size + alignment;
        m_allocated_buffer_pool = static_cast<char*>(ngraph_malloc(allocation_size));
        m_aligned_buffer_pool = m_allocated_buffer_pool;
        size_t mod = size_t(m_aligned_buffer_pool) % alignment;
        if (mod != 0)
        {
            m_aligned_buffer_pool += (alignment - mod);
        }
    }
}

runtime::HostTensor::HostTensor(const ngraph::element::Type& element_type,
                                const Shape& shape,
                                const string& name)
    : HostTensor(element_type, shape, nullptr, name)
{
}

runtime::HostTensor::HostTensor(const ngraph::element::Type& element_type, const Shape& shape)
    : HostTensor(element_type, shape, nullptr, "")
{
}

runtime::HostTensor::HostTensor(const ngraph::element::Type& element_type,
                                const Shape& shape,
                                void* memory_pointer)
    : HostTensor(element_type, shape, memory_pointer, "")
{
}

runtime::HostTensor::~HostTensor()
{
    if (m_allocated_buffer_pool != nullptr)
    {
        ngraph_free(m_allocated_buffer_pool);
    }
}

char* runtime::HostTensor::get_data_ptr()
{
    return m_aligned_buffer_pool;
}

const char* runtime::HostTensor::get_data_ptr() const
{
    return m_aligned_buffer_pool;
}

void runtime::HostTensor::write(const void* source, size_t n)
{
    event::Duration d1("write", "HostTensor");

    if (n != m_buffer_size)
    {
        throw out_of_range("partial tensor write not supported");
    }
    char* target = get_data_ptr();
    memcpy(target, source, n);
}

void runtime::HostTensor::read(void* target, size_t n) const
{
    event::Duration d1("read", "HostTensor");
    if (n != m_buffer_size)
    {
        throw out_of_range("partial tensor read access not supported");
    }
    const char* source = get_data_ptr();
    memcpy(target, source, n);
}
