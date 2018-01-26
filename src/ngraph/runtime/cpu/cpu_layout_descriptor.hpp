// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#pragma once

#include <cstdint>

#include <mkldnn_types.h>

#include "ngraph/common.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/descriptor/layout/tensor_view_layout.hpp"
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
                                 const AxisVector& tv_axis_order)
                    : TensorViewLayout(tv)
                    , axis_order(tv_axis_order)
                    , offset(0)
                    , size(ngraph::shape_size(tv.get_tensor_view_type()->get_shape()))
                    , mkldnn_format(mkldnn_format_undef)
                {
                    if (tv_axis_order == Native2DAxisOrder ||
                        tv_axis_order == Native4DAxisOrder) {
                        strides = ngraph::row_major_strides(get_shape());
                    }
                    else
                    {
                        throw ngraph_error("Axis ordering not handled yet");
                    }
                }

                size_t get_size() override { return size; }
                size_t get_offset() const { return offset; }
                size_t get_index_offset(const std::vector<size_t>& indices) override;

                const Strides& get_strides() const override { return strides; }
                bool operator==(const TensorViewLayout& other) const override;

                mkldnn_memory_format_t get_mkldnn_format() const
                {
                    return mkldnn_format;
                }

                static const AxisVector Native2DAxisOrder;
                static const AxisVector Native4DAxisOrder;
                static const AxisVector CHWNAxisOrder;
            private:
                AxisVector axis_order;
                Strides strides;
                size_t offset;
                size_t size;

                // Numeric backend-specific fields 
                mkldnn_memory_format_t mkldnn_format;
            };
        }
    }
}
