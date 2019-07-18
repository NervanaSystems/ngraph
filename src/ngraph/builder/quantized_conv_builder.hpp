//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/quantized_convolution.hpp"
#include "quantization_utils.hpp"

namespace ngraph
{
    namespace builder
    {
        std::shared_ptr<Node>
            QuantizedConvolutionBuilder(const std::shared_ptr<Node>& input,
                                        const std::shared_ptr<Node>& filters,
                                        const Strides& window_movement_strides,
                                        const Strides& window_dilation_strides,
                                        const CoordinateDiff& padding_below,
                                        const CoordinateDiff& padding_above,
                                        const Strides& data_dilation_strides,
                                        const std::shared_ptr<Node>& min_input,
                                        const std::shared_ptr<Node>& max_input,
                                        const std::shared_ptr<Node>& min_filter,
                                        const std::shared_ptr<Node>& max_filter,
                                        const std::shared_ptr<Node>& min_output,
                                        const std::shared_ptr<Node>& max_output,
                                        const ngraph::element::Type& output_type,
                                        const ngraph::AxisSet& input_axes = ngraph::AxisSet{},
                                        const ngraph::AxisSet& filter_axes = ngraph::AxisSet{},
                                        const ngraph::AxisSet& output_axes = ngraph::AxisSet{});
    }
}
