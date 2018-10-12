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

#pragma once

#include <tuple>

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/op.hpp"

namespace ngraph
{
    Shape infer_convolution_output_item_shape(const Node* node,
                                              const Shape& data_shape,
                                              const Strides& data_dilation,
                                              const CoordinateDiff& data_padding_below,
                                              const CoordinateDiff& data_padding_above,
                                              const Shape& filter_shape,
                                              const Strides& filter_strides,
                                              const Strides& filter_dilation);

    std::tuple<element::Type, Shape>
        infer_convolution_forward(const Node* node,
                                  element::Type et_batch,
                                  element::Type et_filters,
                                  const Shape& data_batch_shape,
                                  const Strides& data_dilation,
                                  const CoordinateDiff& data_padding_below,
                                  const CoordinateDiff& data_padding_above,
                                  const Shape& filters_shape,
                                  const Strides& filter_strides,
                                  const Strides& filter_dilation);
}
