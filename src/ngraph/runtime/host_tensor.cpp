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
#include "ngraph/evaluator_tensor.hpp"
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

namespace
{
    // Wraps an existing HostTensor
    class HostTensorEvaluatorTensor : public runtime::HostTensor::HostEvaluatorTensor
    {
    public:
        HostTensorEvaluatorTensor(const element::Type& element_type,
                                  const PartialShape& partial_shape)
            : HostEvaluatorTensor(element_type, partial_shape)
        {
        }
        HostTensorEvaluatorTensor(shared_ptr<runtime::HostTensor> host_tensor)
            : HostEvaluatorTensor(host_tensor->get_element_type(), host_tensor->get_partial_shape())
            , m_host_tensor(host_tensor)
        {
        }
        void* get_data_ptr() override
        {
            if (!m_host_tensor)
            {
                NGRAPH_CHECK(m_element_type.is_static(),
                             "Attempt to create host tensor with a dynamic element type: ",
                             m_element_type);
                NGRAPH_CHECK(m_partial_shape.is_static(),
                             "Attempt to create a host tensor with a dynamic shape: ",
                             m_partial_shape);
                m_host_tensor =
                    make_shared<runtime::HostTensor>(m_element_type, m_partial_shape.get_shape());
            }
            return m_host_tensor->get_data_ptr();
        }
        shared_ptr<runtime::HostTensor> get_host_tensor() override { return m_host_tensor; }
    private:
        shared_ptr<runtime::HostTensor> m_host_tensor;
    };
}

runtime::HostTensor::HostEvaluatorTensorPtr
    runtime::HostTensor::create_evaluator_tensor(std::shared_ptr<runtime::HostTensor> host_tensor)
{
    return make_shared<HostTensorEvaluatorTensor>(host_tensor);
}

runtime::HostTensor::HostEvaluatorTensorPtr
    runtime::HostTensor::create_evaluator_tensor(const element::Type& element_type,
                                                 const PartialShape& partial_shape)
{
    return make_shared<HostTensorEvaluatorTensor>(element_type, partial_shape);
}
