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

            AxisVector LayoutDescriptor::create_native_axis_order(size_t rank)
            {
                AxisVector native_axis_order(rank);
                std::iota(native_axis_order.begin(), native_axis_order.end(), 0);
                return native_axis_order;
            }

            LayoutDescriptor::LayoutDescriptor(const ngraph::descriptor::TensorView& tv,
                                               const AxisVector& tv_axis_order)
                : TensorViewLayout(tv)
                , axis_order(tv_axis_order)
                , offset(0)
                , size(ngraph::shape_size(tv.get_tensor_view_type()->get_shape()))
                , m_mkldnn_layout(false)
                , m_mkldnn_format(mkldnn::memory::format::format_undef)
                , m_mkldnn_md(LayoutDescriptor::DummyDesc)
            {
                auto shape = get_shape();
                size_t s = 1;

                if (tv_axis_order.size() != shape.size())
                {
                    throw ngraph_error("Axis order is incomplete");
                }

                for (auto it = tv_axis_order.crbegin(); it != tv_axis_order.crend(); it++)
                {
                    if (*it >= shape.size())
                    {
                        throw ngraph_error("Axis is out of bounds");
                    }

                    strides.emplace_back(s);
                    s *= shape[*it];
                }
                std::reverse(strides.begin(), strides.end());
            }

            void LayoutDescriptor::set_axis_order(const AxisVector& perm) { axis_order = perm; }
            size_t LayoutDescriptor::get_index_offset(const std::vector<size_t>& indices)
            {
                if (indices.size() != strides.size())
                {
                    throw ngraph_error("Indices have incorrect rank");
                }
                size_t result = 0;
                for (int i = 0; i < indices.size(); i++)
                {
                    result += strides[i] + indices[i];
                }
                return result;
            }

            bool LayoutDescriptor::
                operator==(const ngraph::descriptor::layout::TensorViewLayout& other) const
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
                    if (!m_mkldnn_layout)
                    {
                        return false;
                    }
                    return runtime::cpu::mkldnn_utils::compare_mkldnn_mds(m_mkldnn_md, p_other->get_mkldnn_md());
                }

                if (strides != p_other->strides)
                {
                    return false;
                }

                if (offset != p_other->offset)
                {
                    return false;
                }

                return true;
            }
        }
    }
}
