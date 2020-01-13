//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <algorithm>
#include <memory>
#include <vector>

#include "default_opset.hpp"
#include "ngraph/node.hpp"
#include "utils/common.hpp"

namespace
{
    int64_t get_valid_array_idx(int64_t idx, int64_t last_idx)
    {
        return (idx >= 0) ? std::min(idx, last_idx) : std::max<int64_t>(0, last_idx + idx);
    }
}

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector slice(const Node& node)
                {
                    std::shared_ptr<ngraph::Node> data = node.get_ng_inputs().at(0);
                    Shape data_shape = data->get_shape();
                    const auto data_rank = data_shape.size();

                    auto starts = node.get_attribute_value<std::vector<int64_t>>("starts");
                    auto ends = node.get_attribute_value<std::vector<int64_t>>("ends");

                    auto axes = node.get_attribute_value<std::vector<int64_t>>(
                        "axes", common::get_monotonic_range<int64_t>(data_shape.size()));

                    Shape lower_bounds(data_rank);
                    Shape upper_bounds = data_shape;

                    for (size_t idx = 0; idx < axes.size(); ++idx)
                    {
                        size_t axis = axes.at(idx);
                        lower_bounds.at(axis) =
                            get_valid_array_idx(starts.at(idx), data_shape.at(axis));
                        upper_bounds.at(axis) =
                            get_valid_array_idx(ends.at(idx), data_shape.at(axis));
                    }

                    // Check for cases when start is greater than end and change them to "empty"
                    // slice.
                    for (size_t idx = 0; idx < lower_bounds.size(); ++idx)
                    {
                        if (lower_bounds.at(idx) > upper_bounds.at(idx))
                        {
                            upper_bounds.at(idx) = lower_bounds.at(idx);
                        }
                    }

                    const auto begin = ngraph::op::Constant::create(
                        element::i64, Shape{lower_bounds.size()}, lower_bounds);
                    const auto end = ngraph::op::Constant::create(
                        element::i64, Shape{upper_bounds.size()}, upper_bounds);
                    const auto strides = ngraph::op::Constant::create(
                        element::i64, Shape{data_rank}, std::vector<int64_t>(data_rank, 1));

                    return {std::make_shared<default_opset::StridedSlice>(
                        data,
                        begin,
                        end,
                        strides,
                        std::vector<int64_t>(data_rank, 0),
                        std::vector<int64_t>(data_rank, 0))};
                }

            } // namespace set_1

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
