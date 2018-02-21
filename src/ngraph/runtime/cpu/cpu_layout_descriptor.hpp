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

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <mkldnn.hpp>

#include "ngraph/common.hpp"
#include "ngraph/descriptor/layout/tensor_view_layout.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/types/type.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            class LayoutDescriptor : public ngraph::descriptor::layout::TensorViewLayout
            {
            public:
                LayoutDescriptor(const ngraph::descriptor::TensorView& tv,
                                 const AxisVector& tv_axis_order);
                ~LayoutDescriptor() {}
                size_t get_size() override { return m_size; }
                size_t get_offset() const { return m_offset; }
                size_t get_index_offset(const std::vector<size_t>& indices) override;

                const Strides& get_strides() const override { return m_strides; }
                bool operator==(const TensorViewLayout& other) const override;

                void set_mkldnn_format(const mkldnn::memory::format& format)
                {
                    m_mkldnn_format = format;
                }
                mkldnn::memory::format get_mkldnn_format() const { return m_mkldnn_format; }
                const AxisVector& get_axis_order() const { return m_axis_order; }
                static const AxisVector s_native_2d_axis_order;
                static const AxisVector s_native_4d_axis_order;
                static const AxisVector s_chwn_axis_order;
                static AxisVector create_native_axis_order(size_t rank);

            private:
                AxisVector m_axis_order;
                Strides m_strides;
                size_t m_offset;
                size_t m_size;

                // Numeric backend-specific fields
                mkldnn::memory::format m_mkldnn_format;
            };

            using LayoutDescriptors =
                std::vector<std::shared_ptr<ngraph::runtime::cpu::LayoutDescriptor>>;
        }
    }
}
