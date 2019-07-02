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

namespace ngraph
{
    namespace builder
    {
        namespace quantization
        {
            std::shared_ptr<Node>
                QuantizedLinearConvolution(const std::shared_ptr<Node>& input,
                                           const std::shared_ptr<Node>& filter,
                                           const Strides& window_movement_strides,
                                           const Strides& window_dilation_strides,
                                           const CoordinateDiff& padding_below,
                                           const CoordinateDiff& padding_above,
                                           const Strides& data_dilation_strides,
                                           const std::shared_ptr<Node>& input_scale,
                                           const std::shared_ptr<Node>& filter_scale,
                                           const std::shared_ptr<Node>& output_scale);

            std::shared_ptr<Node>
                QuantizedLinearConvolution(const std::shared_ptr<Node>& input,
                                           const std::shared_ptr<Node>& filter,
                                           const Strides& window_movement_strides,
                                           const Strides& window_dilation_strides,
                                           const CoordinateDiff& padding_below,
                                           const CoordinateDiff& padding_above,
                                           const Strides& data_dilation_strides,
                                           const std::shared_ptr<Node>& input_scale,
                                           const std::shared_ptr<Node>& input_zero_point,
                                           const std::shared_ptr<Node>& filter_scale,
                                           const std::shared_ptr<Node>& filter_zero_point,
                                           const std::shared_ptr<Node>& output_scale,
                                           const std::shared_ptr<Node>& output_zero_point);

            std::shared_ptr<Node>
                QuantizedLinearConvolutionBias(const std::shared_ptr<Node>& input,
                                               const std::shared_ptr<Node>& filter,
                                               const std::shared_ptr<Node>& bias,
                                               const Strides& window_movement_strides,
                                               const Strides& window_dilation_strides,
                                               const CoordinateDiff& padding_below,
                                               const CoordinateDiff& padding_above,
                                               const Strides& data_dilation_strides,
                                               const std::shared_ptr<Node>& input_scale,
                                               const std::shared_ptr<Node>& filter_scale,
                                               const std::shared_ptr<Node>& output_scale);

            std::shared_ptr<Node> QuantizedConvInteger(const std::shared_ptr<Node>& input,
                                                       const std::shared_ptr<Node>& filter,
                                                       const Strides& window_movement_strides,
                                                       const Strides& window_dilation_strides,
                                                       const CoordinateDiff& padding_below,
                                                       const CoordinateDiff& padding_above,
                                                       const Strides& data_dilation_strides);

            std::shared_ptr<Node>
                QuantizedConvInteger(const std::shared_ptr<Node>& input,
                                     const std::shared_ptr<Node>& filter,
                                     const Strides& window_movement_strides,
                                     const Strides& window_dilation_strides,
                                     const CoordinateDiff& padding_below,
                                     const CoordinateDiff& padding_above,
                                     const Strides& data_dilation_strides,
                                     const std::shared_ptr<Node>& input_zero_point,
                                     const std::shared_ptr<Node>& filter_zero_point);
        }
    }
}
