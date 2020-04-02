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
#include "gather.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/constant.hpp"
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
            namespace set_10
            {
                namespace
                {
                    std::vector<int64_t> axes_to_mask(const std::vector<int64_t>& axes)
                    {
                        std::vector<int64_t> mask(
                            *std::max_element(std::begin(axes), std::end(axes)) + 1, 1);
                        for (int i = 0; i < axes.size(); ++i)
                        {
                            mask[axes[i]] = 0;
                        }
                        return mask;
                    }
                }

                NodeVector slice(const Node& node)
                {
                    NodeVector inputs{node.get_ng_inputs()};
                    const auto data = inputs.at(0);

                    auto starts = inputs.at(1);
                    auto ends = inputs.at(2);

                    // Slice is calculated over all axes as default
                    std::shared_ptr<ngraph::Node> axes;
                    if (inputs.size() >= 4) // axes input provided
                    {
                        axes = inputs.at(3);
                        NGRAPH_CHECK(axes->is_constant(), "Axes input must be constant");
                    }
                    else
                    {
                        const size_t data_rank =
                            data->get_output_partial_shape(0).rank().get_length();
                        axes = default_opset::Constant::create(
                            element::i64,
                            {data_rank},
                            common::get_monotonic_range<int64_t>(data_rank));
                    }

                    const bool is_steps_provided = inputs.size() == 5;
                    std::shared_ptr<ngraph::Node> steps = nullptr;
                    if (is_steps_provided)
                    {
                        steps = inputs.at(4);
                    }

                    const auto axes_const = as_type_ptr<default_opset::Constant>(axes);
                    const auto axes_vec = axes_const->cast_vector<int64_t>();

                    // if axes have not growing elements, order of starts, ends, steps must adjusted
                    if (!std::is_sorted(axes_vec.begin(), axes_vec.end()))
                    {
                        std::vector<int64_t> new_order(axes_vec.size());
                        for (int i = 0; i < new_order.size(); ++i)
                        {
                            new_order[axes_vec[i]] = i;
                        }
                        const auto new_order_const = default_opset::Constant::create(
                            element::i64, {new_order.size()}, new_order);
                        const auto gather_axis =
                            default_opset::Constant::create(element::i64, {}, {0});
                        starts = std::make_shared<default_opset::Gather>(
                            starts, new_order_const, gather_axis);
                        ends = std::make_shared<default_opset::Gather>(
                            ends, new_order_const, gather_axis);
                        if (is_steps_provided)
                        {
                            steps = std::make_shared<default_opset::Gather>(
                                steps, new_order_const, gather_axis);
                        }
                    }

                    const auto begin_end_mask = axes_to_mask(axes_vec);
                    if (is_steps_provided)
                    {
                        return {std::make_shared<default_opset::StridedSlice>(
                            data, starts, ends, steps, begin_end_mask, begin_end_mask)};
                    }
                    else
                    {
                        return {std::make_shared<default_opset::StridedSlice>(
                            data, starts, ends, begin_end_mask, begin_end_mask)};
                    }
                }
            } // namespace set_10
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

                    const auto begin = default_opset::Constant::create(
                        element::i64, Shape{lower_bounds.size()}, lower_bounds);
                    const auto end = default_opset::Constant::create(
                        element::i64, Shape{upper_bounds.size()}, upper_bounds);
                    const auto strides = default_opset::Constant::create(
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
