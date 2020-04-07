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
                    std::vector<int64_t> axes_to_mask(const std::vector<int64_t>& axes,
                                                      uint64_t slice_indices_length)
                    {
                        std::vector<int64_t> mask(slice_indices_length, 1);
                        for (auto axis : axes)
                        {
                            mask[axis] = 0;
                        }
                        return mask;
                    }

                    std::shared_ptr<ngraph::Node>
                        adjust_indices_if_needed(const std::shared_ptr<ngraph::Node>& input,
                                                 const std::vector<int64_t>& axes,
                                                 uint64_t slice_indices_length,
                                                 int64_t value)
                    {
                        const bool are_axes_sorted = std::is_sorted(axes.begin(), axes.end());

                        const auto input_shape = input->get_output_partial_shape(0);
                        // if length of slice indices vector is known
                        if (input_shape.rank().is_static() &&
                            input_shape.rank().get_length() == 1 && input_shape[0].is_static())
                        {
                            if (input_shape[0].get_length() >= slice_indices_length &&
                                are_axes_sorted)
                            {
                                // adjusting indices is not needed
                                return input;
                            }
                        }
                        // Handle a case when starts/ends/steps lengths are less than provided axes
                        // in order to ensure compatibility with `StridedSlice:v1` interface
                        // Example:
                        // data_shape: {3, 3, 3, 3}
                        // starts: [1, 1] - after extending --> [0, 0, 1, 1]
                        // ends: [2, 2] - after extending --> [0, 0, 2, 2]
                        // steps : [0, 1] - after extending --> [1, 1, 0, 1] (`1` is neutral as a
                        // strides value)
                        // axes: [2, 3] - apply slice values to 2 and 3 dimension of input data
                        // expected_output_shape: {3, 3, 1, 1}
                        OutputVector adjusted_indices(slice_indices_length);
                        std::vector<int64_t> target_axes(axes);
                        const auto gather_axis =
                            default_opset::Constant::create(element::i64, {}, {0});

                        int added_indices_number = 0;
                        for (int i = 0; i < slice_indices_length; ++i)
                        {
                            if (std::find(std::begin(axes), std::end(axes), i) == axes.end())
                            {
                                adjusted_indices[i] =
                                    default_opset::Constant::create(element::i64, {1}, {value});
                                target_axes.insert(std::next(target_axes.begin(), i), i);
                                ++added_indices_number;
                            }
                            else
                            {
                                adjusted_indices[i] = std::make_shared<default_opset::Gather>(
                                    input,
                                    default_opset::Constant::create(
                                        element::i64, {1}, {i - added_indices_number}),
                                    gather_axis);
                            }
                        }

                        if (!are_axes_sorted)
                        {
                            OutputVector indices_tmp(adjusted_indices);
                            for (int i = 0; i < target_axes.size(); ++i)
                            {
                                adjusted_indices[target_axes[i]] = indices_tmp[i];
                            }
                        }

                        return std::make_shared<default_opset::Concat>(adjusted_indices, 0);
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
                        const auto data_rank = data->get_output_partial_shape(0).rank();
                        NGRAPH_CHECK(data_rank.is_static(),
                                     "Data rank must be static when axes input is not provided");
                        const size_t data_rank_value = data_rank.get_length();
                        axes = default_opset::Constant::create(
                            element::i64,
                            {data_rank_value},
                            common::get_monotonic_range<int64_t>(data_rank_value));
                    }

                    const auto axes_const = as_type_ptr<default_opset::Constant>(axes);
                    auto axes_vec = axes_const->cast_vector<int64_t>();
                    const uint64_t slice_indices_length =
                        *std::max_element(std::begin(axes_vec), std::end(axes_vec)) + 1;
                    const auto begin_end_mask = axes_to_mask(axes_vec, slice_indices_length);

                    std::shared_ptr<ngraph::Node> steps = nullptr;
                    if (inputs.size() == 5) // steps input provided
                    {
                        steps = inputs.at(4);
                    }
                    else
                    {
                        steps = default_opset::Constant::create(
                            element::i64,
                            {slice_indices_length},
                            std::vector<int64_t>(slice_indices_length, 1));
                    }

                    starts = adjust_indices_if_needed(starts, axes_vec, slice_indices_length, 0);
                    ends = adjust_indices_if_needed(ends, axes_vec, slice_indices_length, 0);
                    steps = adjust_indices_if_needed(steps, axes_vec, slice_indices_length, 1);

                    return {std::make_shared<default_opset::StridedSlice>(
                        data, starts, ends, steps, begin_end_mask, begin_end_mask)};
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
