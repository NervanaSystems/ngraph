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

#include <iterator>
#include <numeric>
#include <vector>

#include "ngraph/axis_vector.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/reshape.hpp"

#include "broadcasting.hpp"
#include "reshape.hpp"

/// \brief Calculate output shape of numpy - style broadcast operation.
///        https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#general-broadcasting-rules
///
/// \param left_shape Shape of first input tensor.
/// \param right_shape Shape of the second input tensor.
/// \return Shape of the output tensor and full shape of input tensors.
std::vector<ngraph::Shape> get_numpy_broadcast_shape(ngraph::Shape left_shape,
                                                     ngraph::Shape right_shape)
{
    ngraph::Shape output_shape;
    auto rank_left = left_shape.size();
    auto rank_right = right_shape.size();
    auto max_rank = std::max(rank_left, rank_right);

    for (auto i = 0; i < (max_rank - rank_left); ++i)
    {
        left_shape.insert(std::begin(left_shape), 1);
    }
    for (auto i = 0; i < (max_rank - rank_right); ++i)
    {
        right_shape.insert(std::begin(right_shape), 1);
    }
    for (auto index = 0; index < max_rank; ++index)
    {
        output_shape.push_back(std::max(left_shape.at(index), right_shape.at(index)));
    }

    return {output_shape, left_shape, right_shape};
}

/// \brief      Broadcast input node.
///
/// \note       The source shape does not have to be the actual shape of input node. However
///             it should be a superset of it (containing it as a continuous subset). This implies
///             we may expand the number of axes of input node.
///
/// \param[in]  node          The input Node to be broadcasted.
/// \param[in]  output_shape  The output shape.
/// \param[in]  source_shape  The source shape from which we want to broadcast input node.
///
/// \return     The boroadcasted Node.
///
static std::shared_ptr<ngraph::Node> broadcast(const std::shared_ptr<ngraph::Node>& node,
                                               const ngraph::Shape& output_shape,
                                               const ngraph::Shape& source_shape)
{
    ngraph::AxisVector broadcast_axes;
    ngraph::Shape squeezed_shape;
    // Positions of axes which have length of 1 are needed to calculate broadcast_axes
    // for nGraph broadcast operation. We need to remove all ones from source shape
    // to avoid broadcasting axis conflict.
    for (std::size_t index = 0; index < output_shape.size(); ++index)
    {
        if (source_shape.at(index) == 1)
        {
            broadcast_axes.push_back(index);
        }
        else
        {
            squeezed_shape.push_back(source_shape.at(index));
        }
    }

    // Remove axes which have length of 1 from source shape
    auto broadcasted_node = std::make_shared<ngraph::op::Reshape>(
        node,
        ngraph::onnx_import::reshape::get_default_axis_vector(node->get_shape().size()),
        squeezed_shape);

    return std::make_shared<ngraph::op::Broadcast>(broadcasted_node, output_shape, broadcast_axes);
}

namespace ngraph
{
    namespace onnx_import
    {
        NodeVector
            numpy_style_broadcast_for_binary_operation(const std::shared_ptr<ngraph::Node>& left,
                                                       const std::shared_ptr<ngraph::Node>& right)
        {
            const auto& left_shape = left->get_shape();
            const auto& right_shape = right->get_shape();
            const auto& numpy_shapes = get_numpy_broadcast_shape(left_shape, right_shape);
            auto output_shape = numpy_shapes.at(0);
            auto left_full_shape = numpy_shapes.at(1);
            auto right_full_shape = numpy_shapes.at(2);

            return {broadcast(left, output_shape, left_full_shape),
                    broadcast(right, output_shape, right_full_shape)};
        }

        NodeVector
            numpy_style_broadcast_for_matmul_operation(const std::shared_ptr<ngraph::Node>& left,
                                                       const std::shared_ptr<ngraph::Node>& right)
        {
            const auto& left_shape = left->get_shape();
            const auto& right_shape = right->get_shape();
            // Broadcast only _stack of matrices_ axes.
            const auto& numpy_shapes = get_numpy_broadcast_shape(
                Shape{std::begin(left_shape), std::next(std::end(left_shape), -2)},
                Shape{std::begin(right_shape), std::next(std::end(right_shape), -2)});

            // Prepare tensors output shapes with broadcasted _stack of matrices_ axes.
            auto left_output_shape = numpy_shapes.at(0);
            auto right_output_shape = numpy_shapes.at(0);
            // Append the last two axes original dimensions.
            left_output_shape.insert(std::end(left_output_shape),
                                     std::next(std::begin(left_shape), left_shape.size() - 2),
                                     std::end(left_shape));
            right_output_shape.insert(std::end(right_output_shape),
                                      std::next(std::begin(right_shape), right_shape.size() - 2),
                                      std::end(right_shape));

            auto left_full_shape = numpy_shapes.at(1);
            auto right_full_shape = numpy_shapes.at(2);
            // Append the last two axes original dimensions.
            left_full_shape.insert(std::end(left_full_shape),
                                   std::next(std::begin(left_shape), left_shape.size() - 2),
                                   std::end(left_shape));
            right_full_shape.insert(std::end(right_full_shape),
                                    std::next(std::begin(right_shape), right_shape.size() - 2),
                                    std::end(right_shape));

            return {broadcast(left, left_output_shape, left_full_shape),
                    broadcast(right, right_output_shape, right_full_shape)};
        }

        NodeVector
            legacy_style_broadcast_for_binary_operation(const std::shared_ptr<ngraph::Node>& left,
                                                        const std::shared_ptr<ngraph::Node>& right,
                                                        std::size_t start_match_axis)
        {
            const auto& left_shape = left->get_shape();
            const auto& right_shape = right->get_shape();

            bool dimensions_identical = (left_shape == right_shape);
            if (dimensions_identical)
            {
                return {left, right};
            }

            // Prepare new shape of right operand for broadcasting
            // Remove dimensions with length=1 from back
            auto new_right_shape = right_shape;
            for (int dimension = new_right_shape.size() - 1; dimension >= 0; --dimension)
            {
                if (new_right_shape[dimension] == 1)
                {
                    new_right_shape.pop_back();
                }
                else
                {
                    break;
                }
            }

            // Find first dimensions at front with length different from 1
            size_t num_ones = 0;
            for (size_t dimension : new_right_shape)
            {
                if (dimension == 1)
                {
                    ++num_ones;
                }
                else
                {
                    break;
                }
            }

            // Remove dimensions with length=1 from front
            new_right_shape.erase(std::begin(new_right_shape),
                                  std::next(std::begin(new_right_shape), num_ones));

            auto reshape_right = std::make_shared<ngraph::op::Reshape>(
                right, reshape::get_default_axis_vector(right_shape.size()), new_right_shape);

            // Move broadcast start axis parameter to right
            start_match_axis += num_ones;

            auto broadcast_right = std::make_shared<ngraph::op::Broadcast>(
                reshape_right,
                left_shape,
                calculate_broadcast_axes(left_shape, new_right_shape, start_match_axis));

            return {left, broadcast_right};
        }

        AxisSet calculate_broadcast_axes(const Shape& output_shape,
                                         const Shape& input_shape,
                                         std::size_t start_match_axis)
        {
            std::vector<size_t> result(output_shape.size() - input_shape.size());
            // Populate the result vector with monotonic increasing series from 0 until
            // output_shape_size, excluding values in range [start_match_axis, start_match_axis + input_shape.size()
            std::iota(std::begin(result), std::begin(result) + start_match_axis, 0);
            std::iota(std::begin(result) + start_match_axis,
                      std::end(result),
                      start_match_axis + input_shape.size());
            return result;
        }

    } // namespace  onnx_import

} // namespace  ngraph
