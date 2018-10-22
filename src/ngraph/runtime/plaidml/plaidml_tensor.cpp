/*******************************************************************************
* Copyright 2018 Intel Corporation
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
    , tensor_{config->dev->allocate(
          to_plaidml(config->ctx, element_type, shape, ConversionUse::FOR_IO))}
    , memory_{memory}
    , memory_size_{memory ? tensor_.get_shape().buffer_size() : 0}
    , is_logically_zero_{memory ? false : true}
{
    m_descriptor->set_tensor_layout(
        std::make_shared<ngraph::descriptor::layout::DenseTensorLayout>(*m_descriptor));
    NGRAPH_DEBUG << "Built PlaidML_Tensor " << this << " memory=" << memory_
                 << " type=" << element_type << " shape=" << shape;
}

void ngraph::runtime::plaidml::PlaidML_Tensor::write(const void* p, size_t tensor_offset, size_t n)
{
    NGRAPH_DEBUG << "Write " << this << " offset=" << tensor_offset << " n=" << n
                 << " is_logically_zero=" << is_logically_zero_;

    // As a special case: if we get a zero-sized write to offset zero, fill the tensor with zero.
    if (n == 0 && tensor_offset == 0)
    {
        NGRAPH_DEBUG << "Logically zeroing tensor " << this;
        is_logically_zero_ = true;
        return;
    }

    bool is_full_write = (tensor_offset == 0 && n == tensor_.get_shape().buffer_size());

    vp::mapping<char> mp;
    if (is_logically_zero_ || is_full_write)
    {
        // In either of these cases, we're completely replacing the existing data.
        mp = tensor_.map(vp::map_for_write);
    }
    else
    {
        // There may be existing non-zero data, and this is a partial buffer write; we need to read the existing data.
        mp = tensor_.map(vp::map_for_update);
    }

    if (is_logically_zero_ && !is_full_write)
    {
        // It's a partial write of a logically-zero buffer, so first, fill the buffer with physical zeros.
        std::fill_n(mp.raw(), tensor_.get_shape().buffer_size(), 0);
    }
    is_logically_zero_ = false;

    const char* src = static_cast<const char*>(p);
    char* dest = mp.raw() + tensor_offset;
    std::copy(src, src + n, dest);
}

void ngraph::runtime::plaidml::PlaidML_Tensor::read(void* p, size_t tensor_offset, size_t n) const
{
    NGRAPH_DEBUG << "Read " << this << " offset=" << tensor_offset << " n=" << n
                 << " is_logically_zero=" << is_logically_zero_;

    char* dest = static_cast<char*>(p);

    if (is_logically_zero_)
    {
        std::fill_n(dest, n, 0);
        return;
    }

    vp::mapping<char> mp = tensor_.map(vp::map_for_read);
    const char* src = mp.raw() + tensor_offset;
    std::copy(src, src + n, dest);
}

void ngraph::runtime::plaidml::PlaidML_Tensor::sync_input()
{
    if (!memory_)
    {
        if (is_logically_zero_)
        {
            NGRAPH_DEBUG << "Flushing logically zero " << this << " to physical memory";
            // The tensor's about to be used for an input, and it's logically zero; we need to
            // write physical zeros to its buffer.
            auto mp = tensor_.map(vp::map_for_write);
            std::fill_n(mp.raw(), tensor_.get_shape().buffer_size(), 0);
        }
        is_logically_zero_ = false;
        return;
    }
    NGRAPH_DEBUG << "Syncing input for tensor " << this;
    write(memory_, 0, memory_size_);
}

void ngraph::runtime::plaidml::PlaidML_Tensor::sync_output()
{
    // The tensor's been used for an output, so it's no longer logically zero.
    is_logically_zero_ = false;

    if (!memory_)
    {
        return;
    }
    NGRAPH_DEBUG << "Syncing output for tensor " << this;
    read(memory_, 0, memory_size_);
}
