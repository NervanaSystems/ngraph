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

#include <tuple>

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"

namespace ngraph
{
    Strides conv_default_strides(const Node* node,
                                 const PartialShape& data_batch_shape,
                                 const PartialShape& filters_shape);

    CoordinateDiff conv_default_padding(const Node* node,
                                        const PartialShape& data_batch_shape,
                                        const PartialShape& filters_shape);

    PartialShape infer_windowed_reduction_output_shape(const Node* node,
                                                       const PartialShape& data_shape,
                                                       const Strides& data_dilation,
                                                       const CoordinateDiff& data_padding_below,
                                                       const CoordinateDiff& data_padding_above,
                                                       const PartialShape& window_shape,
                                                       const Strides& window_strides,
                                                       const Strides& window_dilation,
                                                       bool is_window_all_in_padding_allowed,
                                                       bool ceil_mode = false);

    PartialShape infer_convolution_forward(const Node* node,
                                           const PartialShape& data_batch_shape,
                                           const Strides& data_dilation,
                                           const CoordinateDiff& data_padding_below,
                                           const CoordinateDiff& data_padding_above,
                                           const PartialShape& filters_shape,
                                           const Strides& filter_strides,
                                           const Strides& filter_dilation);

    PartialShape infer_batched_pooling_forward(const Node* node,
                                               const PartialShape& data_batch_shape,
                                               const CoordinateDiff& data_padding_below,
                                               const CoordinateDiff& data_padding_above,
                                               const PartialShape& window_shape,
                                               const Strides& window_strides,
                                               bool is_window_all_in_padding_allowed,
                                               bool ceil_mode = false);

    std::tuple<element::Type, PartialShape, PartialShape>
        infer_batch_norm_forward(const Node* node,
                                 element::Type input_element_type,
                                 element::Type gamma_element_type,
                                 element::Type beta_element_type,
                                 element::Type mean_element_type,
                                 element::Type variance_element_type,
                                 const PartialShape& input_shape,
                                 const PartialShape& gamma_shape,
                                 const PartialShape& beta_shape,
                                 const PartialShape& mean_shape,
                                 const PartialShape& variance_shape);

    std::tuple<element::Type, PartialShape, PartialShape>
        infer_batch_norm_forward(const Node* node,
                                 element::Type input_element_type,
                                 element::Type gamma_element_type,
                                 element::Type beta_element_type,
                                 const PartialShape& input_shape,
                                 const PartialShape& gamma_shape,
                                 const PartialShape& beta_shape);

    void infer_auto_padding(const Shape& image_shape,
                            const Shape& filter_shape,
                            const Strides& filter_strides,
                            const Strides& filter_dilations,
                            const op::PadType pad_type,
                            CoordinateDiff& padding_above,
                            CoordinateDiff& padding_below);

    PartialShape infer_slice_shape(const Node* node,
                                   const PartialShape& input_shape,
                                   const std::vector<int64_t>& lb,
                                   const std::vector<int64_t>& ub,
                                   const std::vector<int64_t>& str,
                                   const AxisSet& lb_mask,
                                   const AxisSet& ub_mask,
                                   const AxisSet& new_axis,
                                   const AxisSet& shrink_mask,
                                   const AxisSet& ellipsis_mask);
}
