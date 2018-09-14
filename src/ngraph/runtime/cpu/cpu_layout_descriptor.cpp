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

#include "cpu_layout_descriptor.hpp"
#include <algorithm>
#include <numeric>
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            const mkldnn::memory::desc
                LayoutDescriptor::DummyDesc(mkldnn::memory::dims(TENSOR_MAX_DIMS),
                                            mkldnn::memory::f32,
                                            mkldnn::memory::format::format_undef);

            LayoutDescriptor::LayoutDescriptor(const ngraph::descriptor::Tensor& tv)
                : TensorLayout(tv)
                , m_offset(0)
                , m_mkldnn_md(LayoutDescriptor::DummyDesc)
            {
                auto shape = get_shape();
                size_t s = 1;

                for (size_t i = 0; i < shape.size(); i++)
                {
                    m_strides.emplace_back(s);
                    s *= shape[shape.size() - (i + 1)];
                }
                std::reverse(m_strides.begin(), m_strides.end());
                m_mkldnn_memory_size = shape_size(tv.get_shape()) * tv.get_element_type().size();
            }

            size_t LayoutDescriptor::get_index_offset(const std::vector<size_t>& indices)
            {
                if (indices.size() != m_strides.size())
                {
                    throw ngraph_error("Indices have incorrect rank");
                }
                size_t result = 0;
                for (int i = 0; i < indices.size(); i++)
                {
                    result += m_strides[i] * indices[i];
                }
                return result;
            }

            bool LayoutDescriptor::
                operator==(const ngraph::descriptor::layout::TensorLayout& other) const
            {
                const LayoutDescriptor* p_other = dynamic_cast<const LayoutDescriptor*>(&other);
                if (!p_other)
                {
                    return false;
                }

                if (get_element_type() != p_other->get_element_type())
                {
                    return false;
                }

                if (p_other->is_mkldnn_layout())
                {
                    if (!is_mkldnn_layout())
                    {
                        return false;
                    }
                    return runtime::cpu::mkldnn_utils::compare_mkldnn_mds(m_mkldnn_md,
                                                                          p_other->get_mkldnn_md());
                }

                if (m_strides != p_other->m_strides)
                {
                    return false;
                }

                if (m_offset != p_other->m_offset)
                {
                    return false;
                }

                return true;
            }

            void LayoutDescriptor::set_mkldnn_md(const mkldnn::memory::desc md)
            {
                m_mkldnn_md = md;

                // Since MKLDNN could internally pad the tensor to make blocked layouts
                // we need to compute MKLDNN memory requirement based on its memory desc
                // http://intel.github.io/mkl-dnn/understanding_memory_formats.html
                try
                {
                    auto mem_prim_desc =
                        mkldnn::memory::primitive_desc(md, mkldnn_utils::global_cpu_engine);
                    m_mkldnn_memory_size = mem_prim_desc.get_size();
                }
                catch (const mkldnn::error& e)
                {
                    throw ngraph_error(
                        "error in computing mkldnn memory size from memory primitive desc: " +
                        e.message);
                }
            }
        }
    }
}
