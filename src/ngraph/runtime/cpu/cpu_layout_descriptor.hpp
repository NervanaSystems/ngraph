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

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <dnnl.hpp>

#include "cpu_backend_visibility.h"
#include "ngraph/descriptor/layout/tensor_layout.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            class CPU_BACKEND_API LayoutDescriptor : public ngraph::descriptor::layout::TensorLayout
            {
            public:
                LayoutDescriptor(const ngraph::descriptor::Tensor& tv);
                ~LayoutDescriptor() override {}
                virtual size_t get_allocated_size() override { return m_buffer_size; }
                size_t get_offset() const { return m_offset; }
                size_t get_index_offset(const std::vector<size_t>& indices) override;

                Strides get_strides() const override { return m_strides; }
                void set_strides(Strides& strides) { m_strides = strides; }
                bool operator==(const TensorLayout& other) const override;

                const dnnl::memory::desc& get_dnnl_md() const { return m_dnnl_md; }
                void set_dnnl_md(const dnnl::memory::desc& md);
                bool is_dnnl_layout() const
                {
                    return static_cast<dnnl::memory::format_kind>(m_dnnl_md.data.format_kind) !=
                           dnnl::memory::format_kind::undef;
                }
                bool is_row_major_layout();

                static const dnnl::memory::desc DummyDesc;

            private:
                // Native row-major layout for now
                Strides m_strides;
                size_t m_offset;

                // For tensor views that can be tracked with DNNL memory
                // descriptors, this holds the physical layout information
                // Otherwise, physical layout is assumed to be in row-major
                // format represented by m_strides
                dnnl::memory::desc m_dnnl_md;
                size_t m_buffer_size;
            };

            typedef std::vector<std::shared_ptr<ngraph::runtime::cpu::LayoutDescriptor>>
                LayoutDescriptorPtrs;
        }
    }
}
