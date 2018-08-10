/*******************************************************************************
 * Copyright 2018 Intel Corporation
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

#include "ngraph/axis_set.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace util
        {
            /**
             * @brief Generate a list of broadcast axes for ngraph++ broadcast.
             * 
             * @details Informally, a broadcast "adds" axes to the input tensor, replicating 
             *          elements from the input tensor as needed to fill the new dimensions.
             *          Function calculate which of the output axes are added in this way.
             * 
             * @param output_shape The new shape for the output tensor.
             * @param input_shape  The shape of input tensor.
             * @param axis         The axis along which we want to replicate elements. The starting
             *                     axis position (0-based) int the output shape from which the 
             *                     current shape of the tensor matches the desired new shape.
             * @return             The indices of added axes.
             */
            ngraph::AxisSet get_broadcast_axes(const ngraph::Shape& output_shape,
                                               const ngraph::Shape& input_shape,
                                               const std::size_t start_match_axis);

            /**
             * @brief Generate a list of broadcast axes for ngraph++ broadcast.
             * 
             * @details Informally, a broadcast "adds" axes to the input tensor, replicating 
             *          elements from the input tensor as needed to fill the new dimensions.
             *          Function calculate which of the output axes are added in this way.
             * 
             *          This function will attempt to match shapes, assuming the current shape 
             *          matches the rightmost positions of the desired new shape. This behaviour 
             *          is similar to NumPy's broadcasting.
             * 
             * @param output_shape The new shape for the output tensor.
             * @param input_shape  The shape of input tensor.
             * @return             The indices of added axes.
             */
            ngraph::AxisSet get_broadcast_axes(const ngraph::Shape& output_shape,
                                               const ngraph::Shape& input_shape);

        } // namespace  util

    } // namespace  onnx_import

} // namespace  ngraph