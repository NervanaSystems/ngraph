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

#include <cstddef>
#include <iterator>
#include <memory>
#include <vector>

#include "ngraph/coordinate.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/shape.hpp"

#include "matmul.hpp"
#include "utils/broadcasting.hpp"
#include "utils/reshape.hpp"

/// \brief      Slice the sub matrix from 3D input tensor.
///
/// \param[in]  node  The input tensor. Must be 3D.
/// \param[in]  idx   The index on the first axis, at which to slice sub-matrix.
///
/// \return     The node representing sub matrix.
///
static std::shared_ptr<ngraph::Node> get_sub_matrix(const std::shared_ptr<ngraph::Node>& node,
                                                    std::size_t idx)
{
    const ngraph::Shape& shape{node->get_shape()};
    // Below bounds defines the sub_matrix through ranges for each input node axis.
    ngraph::Coordinate lower_bounds(shape.size());
    ngraph::Coordinate upper_bounds = shape;
    // We assume `node` tensor is of rank equal 3, thus we slice the sub-matrix lying in the last
    // two dimensions at index `idx` of first axis.
    lower_bounds.at(0) = idx;
    upper_bounds.at(0) = idx + 1;

    auto sub_matrix = std::shared_ptr<ngraph::Node>{
        std::make_shared<ngraph::op::Slice>(node, lower_bounds, upper_bounds)};
    // Remove first single entry dim.
    return ngraph::onnx_import::reshape::squeeze(sub_matrix);
}

namespace ngraph
{
    namespace onnx_import
    {
        namespace op
        {
            namespace set_1
            {
                NodeVector matmul(const Node& node)
                {
                    const NodeVector& ng_inputs{node.get_ng_inputs()};
                    auto left = std::shared_ptr<ngraph::Node>{ng_inputs.at(0)};
                    auto right = std::shared_ptr<ngraph::Node>{ng_inputs.at(1)};
                    std::size_t left_rank{left->get_shape().size()};
                    std::size_t right_rank{right->get_shape().size()};

                    // First (easy) case:
                    // Multiply two tensors where one of them or both has rank lower equal 2.
                    // This is already internally handled by Ngraph Dot operator.
                    if (left_rank <= 2 || right_rank <= 2)
                    {
                        return {
                            std::make_shared<ngraph::op::Dot>(ng_inputs.at(0), ng_inputs.at(1))};
                    }

                    // Second case:
                    // Multiply two tensors where each of them is rank greater equal 3.

                    // Broadcast input arguments.
                    const NodeVector& broadcasted_nodes =
                        numpy_style_broadcast_for_matmul_operation(left, right);

                    left = broadcasted_nodes.at(0);
                    right = broadcasted_nodes.at(1);
                    const auto& left_shape = left->get_shape();
                    const auto& right_shape = right->get_shape();

                    // Collapse both tensors _stack of matrices_ axes (all except the last two).
                    // This will make easier further dot product calculations.
                    if (left_shape.size() > 3)
                    {
                        left = reshape::collapse(left, 0, left_shape.size() - 3);
                        right = reshape::collapse(right, 0, right_shape.size() - 3);
                    }

                    // Perform multiple small dot products
                    std::size_t groups = left->get_shape().at(0);
                    NodeVector small_dots(groups);

                    for (std::size_t g = 0; g < groups; ++g)
                    {
                        const auto& sliced_left = get_sub_matrix(left, g);
                        const auto& sliced_right = get_sub_matrix(right, g);

                        auto sub_dot = std::make_shared<ngraph::op::Dot>(sliced_left, sliced_right);

                        // Expand sub_dot result with single empty outermost axis, in order to
                        // later concatenate sub_dots at this axis.
                        small_dots.at(g) = reshape::add_empty_axes(sub_dot);
                    }

                    // Concatenate sub_dots on groups axis.
                    auto result = std::make_shared<ngraph::op::Concat>(small_dots, 0);

                    if (left_shape.size() <= 3)
                    {
                        return {result};
                    }
                    // Expand result _stack of matrices_ axes to get expected result shape.
                    else
                    {
                        const Shape& shape{result->get_shape()};
                        Shape result_shape(std::next(std::begin(shape)), std::end(shape));
                        result_shape.insert(
                            std::begin(result_shape),
                            std::begin(left_shape),
                            std::next(std::begin(left_shape), left_shape.size() - 2));
                        return {std::make_shared<ngraph::op::Reshape>(
                            result, reshape::get_default_axis_vector(shape.size()), result_shape)};
                    }
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
