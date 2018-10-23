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

#include "ngraph/runtime/plaidml/plaidml_tensor.hpp"
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/log.hpp"
#include "ngraph/runtime/plaidml/plaidml_translate.hpp"
#include "ngraph/runtime/tensor.hpp"

namespace vp = vertexai::plaidml;

ngraph::runtime::plaidml::PlaidML_Tensor::PlaidML_Tensor(Config* config,
                                                         const ngraph::element::Type& element_type,
                                                         const ngraph::Shape& shape,
                                                         const std::string& name,
                                                         void* memory)
    : Tensor{std::make_shared<ngraph::descriptor::Tensor>(element_type, shape, name)}
    , m_tensor{config->dev->allocate(
          to_plaidml(config->ctx, element_type, shape, ConversionUse::FOR_IO))}
    , m_memory{memory}
    , m_memory_size{memory ? m_tensor.get_shape().buffer_size() : 0}
    , m_is_logically_zero{memory ? false : true}
{
    m_descriptor->set_tensor_layout(
        std::make_shared<ngraph::descriptor::layout::DenseTensorLayout>(*m_descriptor));
    NGRAPH_DEBUG << "Built PlaidML_Tensor " << this << " memory=" << m_memory
                 << " type=" << element_type << " shape=" << shape;
}

void ngraph::runtime::plaidml::PlaidML_Tensor::write(const void* p, size_t tensor_offset, size_t n)
{
    NGRAPH_DEBUG << "Write " << this << " offset=" << tensor_offset << " n=" << n
                 << " is_logically_zero=" << m_is_logically_zero;

    // As a special case: if we get a zero-sized write to offset zero, fill the tensor with zero.
    if (n == 0 && tensor_offset == 0)
    {
        NGRAPH_DEBUG << "Logically zeroing tensor " << this;
        m_is_logically_zero = true;
        return;
    }

    bool is_full_write = (tensor_offset == 0 && n == m_tensor.get_shape().buffer_size());

    vp::mapping<char> mp;
    if (m_is_logically_zero || is_full_write)
    {
        // In either of these cases, we're completely replacing the existing data.
        mp = m_tensor.map(vp::map_for_write);
    }
    else
    {
        // There may be existing non-zero data, and this is a partial buffer write; we need to read
        // the existing data.
        mp = m_tensor.map(vp::map_for_update);
    }

    if (m_is_logically_zero && !is_full_write)
    {
        // It's a partial write of a logically-zero buffer, so first, fill the buffer with physical
        // zeros.
        std::fill_n(mp.raw(), m_tensor.get_shape().buffer_size(), 0);
    }
    m_is_logically_zero = false;

    const char* src = static_cast<const char*>(p);
    char* dest = mp.raw() + tensor_offset;
    std::copy(src, src + n, dest);
}

void ngraph::runtime::plaidml::PlaidML_Tensor::read(void* p, size_t tensor_offset, size_t n) const
{
    NGRAPH_DEBUG << "Read " << this << " offset=" << tensor_offset << " n=" << n
                 << " is_logically_zero=" << m_is_logically_zero;

    char* dest = static_cast<char*>(p);

    if (m_is_logically_zero)
    {
        std::fill_n(dest, n, 0);
        return;
    }

    vp::mapping<char> mp = m_tensor.map(vp::map_for_read);
    const char* src = mp.raw() + tensor_offset;
    std::copy(src, src + n, dest);
}

void ngraph::runtime::plaidml::PlaidML_Tensor::sync_input()
{
    if (!m_memory)
    {
        if (m_is_logically_zero)
        {
            NGRAPH_DEBUG << "Flushing logically zero " << this << " to physical memory";
            // The tensor's about to be used for an input, and it's logically zero; we need to write
            // physical zeros to its buffer.
            auto mp = m_tensor.map(vp::map_for_write);
            std::fill_n(mp.raw(), m_tensor.get_shape().buffer_size(), 0);
        }
        m_is_logically_zero = false;
        return;
    }
    NGRAPH_DEBUG << "Syncing input for tensor " << this;
    write(m_memory, 0, m_memory_size);
}

void ngraph::runtime::plaidml::PlaidML_Tensor::sync_output()
{
    // The tensor's been used for an output, so it's no longer logically zero.
    m_is_logically_zero = false;

    if (!m_memory)
    {
        return;
    }
    NGRAPH_DEBUG << "Syncing output for tensor " << this;
    read(m_memory, 0, m_memory_size);
}
