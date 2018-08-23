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

#include <CPP/topology.hpp>

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/strides.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace intelgpu
        {
            // This implements Convolution nGraph operation
            // nGraph uses channels in this operation but clDNN uses full input data
            void do_convolution_operation(cldnn::topology& topology,
                                          const std::string& input_name,
                                          const Shape& input_shape,
                                          const std::string& filter_name,
                                          const Shape& filter_shape,
                                          const std::string& output_name,
                                          const Shape& output_shape,
                                          const element::Type& output_type,
                                          const CoordinateDiff& pad_below,
                                          const Strides& win_stride,
                                          const Strides& win_dilation,
                                          const Strides& data_dilation,
                                          size_t batch_axis_data,
                                          size_t input_channel_axis_data,
                                          size_t output_channel_axis_result,
                                          const std::string& input_order,
                                          const std::string& filter_order,
                                          const std::string& output_order,
                                          bool reverse_filter);
        }
    }
}
