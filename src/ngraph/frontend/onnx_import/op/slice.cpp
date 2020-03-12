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
#include "ngraph/opsets/opset0.hpp"
#include "ngraph/opsets/opset2.hpp"
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

            namespace set_10
            {
                NodeVector slice(const Node& node)
                {
                    auto get_rank_from_shape = [&](std::shared_ptr<ngraph::Node> shapeof) {
                        auto shapeof_shape = std::make_shared<ngraph::op::ShapeOf>(shapeof);
                        return std::make_shared<ngraph::op::Reshape>(
                            shapeof_shape, ngraph::AxisVector{0}, ngraph::Shape{});
                    };

                    auto ng_range = [&](std::shared_ptr<ngraph::Node> rank_scalar) {
                        return std::make_shared<ngraph::op::Range>(
                            ngraph::op::Constant::create(
                                ngraph::element::i64, ngraph::Shape{}, {0}),
                            rank_scalar,
                            ngraph::op::Constant::create(
                                ngraph::element::i64, ngraph::Shape{}, {1}));
                    };

                    auto get_range_from_shape = [&](std::shared_ptr<ngraph::Node> shapeof) {
                        return ng_range(get_rank_from_shape(shapeof));
                    };

                    auto get_range_from_node = [&](std::shared_ptr<ngraph::Node> node) {
                        auto shapeof = std::make_shared<ngraph::op::ShapeOf>(node);
                        return get_range_from_shape(shapeof);
                    };

                    const NodeVector inputs{node.get_ng_inputs()};

                    std::shared_ptr<ngraph::Node> data = inputs.at(0);
                    std::shared_ptr<ngraph::Node> start = inputs.at(1);
                    std::shared_ptr<ngraph::Node> end = inputs.at(2);

                    std::shared_ptr<ngraph::Node> axes;
                    std::shared_ptr<ngraph::Node> step;

                    auto data_shapeof = std::make_shared<ngraph::op::ShapeOf>(data);
                    auto data_rank = std::make_shared<ngraph::op::ShapeOf>(data_shapeof);
                    if (inputs.size() > 2 && !inputs.at(3)->is_null())
                    {
                        axes = inputs.at(3);
                    }
                    else
                    {
                        auto axes_create = ngraph::op::Constant::create(element::i64, Shape{}, {1});
                        axes = std::make_shared<ngraph::op::DynBroadcast>(
                            axes_create, data_shapeof, get_range_from_node(data_shapeof));
                    }

                    if (inputs.size() > 2 && !inputs.at(4)->is_null())
                    {
                        axes = inputs.at(4);
                    }
                    else
                    {
                        auto step_create = ngraph::op::Constant::create(element::i64, Shape{}, {1});
                        step = std::make_shared<ngraph::op::DynBroadcast>(
                            step_create, data_shapeof, get_range_from_node(data_shapeof));
                    }

                    auto begin_mask_create =
                        ngraph::op::Constant::create(element::i64, Shape{}, {0});

                    auto end_mask_create = ngraph::op::Constant::create(element::i64, Shape{}, {0});

                    auto begin_mask = std::make_shared<ngraph::op::DynBroadcast>(
                        begin_mask_create, data_shapeof, get_range_from_node(data_shapeof));

                    auto end_mask = std::make_shared<ngraph::op::DynBroadcast>(
                        end_mask_create, data_shapeof, get_range_from_node(data_shapeof));

                    return {std::make_shared<ngraph::opset2::StridedSlice>(
                        data, start, end, axes, step, begin_mask, end_mask)};
                }

            } // namespace set_10

        } // namespace op

    } // namespace onnx_import

} // namespace ngraph
