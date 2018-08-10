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

#include <cstddef>
#include <numeric>
#include <vector>

#include "ngraph/frontend/onnx_import/util/broadcasting.hpp"

static std::vector<std::size_t> _get_broadcast_axes(std::size_t output_shape_size,
                                                    std::size_t input_shape_size,
                                                    std::size_t start_match_axis)
{
    // Fill the output vector with monotonic increasing series from 0 till output_shape_size
    // excluding values in range [start_match_axis, start_match_axis + input_shape_size)
    std::vector<size_t> out(output_shape_size - input_shape_size);
    std::iota(out.begin(), out.begin() + start_match_axis, 0);
    std::iota(out.begin() + start_match_axis, out.end(), start_match_axis + input_shape_size);
    return out;
}

namespace ngraph
{
    namespace onnx_import
    {
        namespace util
        {
            ngraph::AxisSet get_broadcast_axes(const ngraph::Shape& output_shape,
                                               const ngraph::Shape& input_shape,
                                               const std::size_t start_match_axis)
            {
                return ngraph::AxisSet{
                    _get_broadcast_axes(output_shape.size(), input_shape.size(), start_match_axis)};
            }

            ngraph::AxisSet get_broadcast_axes(const ngraph::Shape& output_shape,
                                               const ngraph::Shape& input_shape)
            {
                std::size_t start_match_axis = output_shape.size() - input_shape.size();
                return ngraph::AxisSet{
                    _get_broadcast_axes(output_shape.size(), input_shape.size(), start_match_axis)};
            }

        } // namespace  util

    } // namespace  onnx_import

} // namespace  ngraph