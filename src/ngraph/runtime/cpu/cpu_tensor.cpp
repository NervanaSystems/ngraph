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

#include "cpu_tensor.hpp"
#include "ngraph/descriptor/layout/tensor_layout.hpp"
#include "ngraph/except.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/cpu/cpu_executor.hpp"
#include "ngraph/runtime/cpu/cpu_layout_descriptor.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/util.hpp"

using namespace mkldnn;
using namespace ngraph;
using namespace std;

// TODO(jmenon): Refactor all the alignment specifications into
// a single place and allow lower or no alignment when possible

runtime::cpu::CPUTensor::CPUTensor(const ngraph::element::Type& element_type,
                                   const Shape& shape,
                                   void* memory_pointer)
    : runtime::Tensor(std::make_shared<ngraph::descriptor::Tensor>(element_type, shape, ""))
    , buffer(nullptr)
    , aligned_buffer(nullptr)
{
    // TODO(jmenon): A fallback layout should not be needed but is required
    // because of how some unit test functionality is written (ex. 'backprop_derivative')
    // This needs to be removed
    m_descriptor->set_tensor_layout(
        std::make_shared<runtime::cpu::LayoutDescriptor>(*m_descriptor));

    buffer_size = shape_size(shape) * element_type.size();

    if (memory_pointer != nullptr)
    {
        aligned_buffer = static_cast<char*>(memory_pointer);
    }
    else if (buffer_size > 0)
    {
        size_t allocation_size = buffer_size + BufferAlignment;
        auto ptr = ngraph_malloc(allocation_size);
        buffer = static_cast<char*>(ptr);

// GCC major versions below 5 do not implement C++11 std::align
#if !defined(__GNUC__) || __GNUC__ >= 5
        std::align(BufferAlignment, buffer_size, ptr, allocation_size);
#else
        ptr = static_cast<char*>(ptr) + (BufferAlignment - 1);
        ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(ptr) &
                                      ~(uintptr_t(BufferAlignment - 1)));
#endif

        aligned_buffer = static_cast<char*>(ptr);
    }
}

runtime::cpu::CPUTensor::CPUTensor(const ngraph::element::Type& element_type, const Shape& shape)
    : CPUTensor(element_type, shape, nullptr)
{
}

runtime::cpu::CPUTensor::~CPUTensor()
{
    ngraph_free(buffer);
}

char* runtime::cpu::CPUTensor::get_data_ptr()
{
    return aligned_buffer;
}

const char* runtime::cpu::CPUTensor::get_data_ptr() const
{
    return aligned_buffer;
}

void runtime::cpu::CPUTensor::write(const void* source, size_t n)
{
    if (n > buffer_size)
    {
        throw out_of_range("write access past end of tensor");
    }
    char* target = get_data_ptr();
    memcpy(target, source, n);
}

void runtime::cpu::CPUTensor::read(void* target, size_t n) const
{
    if (n > buffer_size)
    {
        throw out_of_range("read access past end of tensor");
    }

    auto tvl = this->get_tensor_layout();
    auto cpu_tvl = dynamic_cast<runtime::cpu::LayoutDescriptor*>(tvl.get());

    auto needs_conversion = [&]() {
        if (!cpu_tvl)
        {
            return false;
        }
        if (!cpu_tvl->is_mkldnn_layout())
        {
            return false;
        }
        if (cpu_tvl->get_size() <= 1)
        {
            return false;
        }
        auto native_md = mkldnn_utils::create_blocked_mkldnn_md(
            this->get_shape(), cpu_tvl->get_strides(), this->get_element_type());
        if (mkldnn_utils::compare_mkldnn_mds(cpu_tvl->get_mkldnn_md(), native_md))
        {
            return false;
        }
        return true;
    };

    if (needs_conversion())
    {
        auto tensor_shape = this->get_shape();
        auto input_desc = cpu_tvl->get_mkldnn_md();
        auto output_desc = mkldnn_utils::create_blocked_mkldnn_md(
            this->get_shape(), cpu_tvl->get_strides(), this->get_element_type());

#if MKLDNN_VERSION_MAJOR < 1
        memory input{{input_desc, executor::global_cpu_engine}, aligned_buffer};
        memory output{{output_desc, executor::global_cpu_engine}, target};
        reorder prim{input, output};
        mkldnn::stream s(mkldnn::stream::kind::eager);
        s.submit({prim}).wait();
#else
        memory input{input_desc, executor::global_cpu_engine, aligned_buffer};
        memory output{output_desc, executor::global_cpu_engine, target};
        reorder prim{input, output};
        mkldnn::stream s(executor::global_cpu_engine);
        prim.execute(s, {{MKLDNN_ARG_SRC, input}, {MKLDNN_ARG_DST, output}});
        s.wait();
#endif
    }
    else
    {
        const char* source = get_data_ptr();
        memcpy(target, source, n);
    }
}

void runtime::cpu::CPUTensor::copy_from(const ngraph::runtime::Tensor& source)
{
    if (get_element_count() != source.get_element_count())
    {
        throw invalid_argument("runtime::cpu::CPUTensor::copy_from element count must match");
    }

    if (get_element_type() != source.get_element_type())
    {
        throw invalid_argument("runtime::cpu::CPUTensor::copy_from element types must match");
    }

    if (auto cpu_source = dynamic_cast<const runtime::cpu::CPUTensor*>(&source))
    {
        auto this_tl =
            dynamic_cast<ngraph::runtime::cpu::LayoutDescriptor*>(this->get_tensor_layout().get());
        auto other_tl =
            dynamic_cast<ngraph::runtime::cpu::LayoutDescriptor*>(source.get_tensor_layout().get());
        if ((this_tl != nullptr) && (other_tl != nullptr) && (*this_tl == *other_tl))
        {
            // Direct copy
            memcpy(get_data_ptr(), cpu_source->get_data_ptr(), get_size_in_bytes());
        }
        else
        {
            // This will copy the data in native/row-major layout
            source.read(get_data_ptr(), get_size_in_bytes());
            // Set default layout
            m_descriptor->set_tensor_layout(
                std::make_shared<runtime::cpu::LayoutDescriptor>(*m_descriptor));
        }
    }
    else
    {
        auto size = get_size_in_bytes();
        AlignedBuffer tmp_buffer{size, static_cast<size_t>(BufferAlignment)};
        source.read(tmp_buffer.get_ptr(), size);
        write(tmp_buffer.get_ptr(), size);
        // Set default layout
        m_descriptor->set_tensor_layout(
            std::make_shared<runtime::cpu::LayoutDescriptor>(*m_descriptor));
    }
}
