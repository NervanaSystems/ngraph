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
            const AxisVector LayoutDescriptor::Native2DAxisOrder{0, 1};
            const AxisVector LayoutDescriptor::Native4DAxisOrder{0, 1, 2, 3};
            const AxisVector LayoutDescriptor::CHWNAxisOrder{1, 2, 3, 0};

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
                , m_size(ngraph::shape_size(tv.get_tensor_view_type()->get_shape()))
                , mkldnn_format(mkldnn::memory::format::format_undef)
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

            void LayoutDescriptor::compute_mkldnn_memory_size(
                std::shared_ptr<const ngraph::TensorViewType> tvt,
                const mkldnn::memory::format& fmt,
                bool is_mkldnn)
            {
                try
                {
                    if (is_mkldnn)
                    {
                        auto mem_desc = mkldnn::memory::desc(
                            mkldnn::memory::dims(tvt->get_shape().begin(), tvt->get_shape().end()),
                            mkldnn_utils::get_mkldnn_data_type(tvt->get_element_type()),
                            fmt);
                        auto mem_prim_desc = mkldnn::memory::primitive_desc(
                            mem_desc, mkldnn_utils::global_cpu_engine);
                        mkldnn_memory_size = mem_prim_desc.get_size();
                    }
                    else
                    {
                        size_t size = 1;
                        for (size_t s : tvt->get_shape())
                        {
                            size *= s;
                        }
                        mkldnn_memory_size = size * tvt->get_element_type().size();
                    }
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
