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

#include <cstddef>
#include <iterator>
#include <numeric>
#include <vector>

#include "broadcasting.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/reshape.hpp"
#include "reshape.hpp"

/// \brief Calculate output shape of numpy - style broadcast operation.
///        https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#general-broadcasting-rules
///
/// \param left_shape Shape of first input tensor.
/// \param right_shape Shape of the second input tensor.
/// \return Shape of the output tensor and full shape of input tensors.
static std::vector<ngraph::Shape> get_numpy_broadcast_shape(ngraph::Shape left_shape,
                                                            ngraph::Shape right_shape)
{
    ngraph::Shape output_shape;
    auto rank_left = left_shape.size();
    auto rank_right = right_shape.size();
    auto max_rank = std::max(rank_left, rank_right);

    // left-pad the left_shape with ones
    left_shape.insert(std::begin(left_shape), max_rank - rank_left, 1);
    // left-pad the right_shape with ones
    right_shape.insert(std::begin(right_shape), max_rank - rank_right, 1);

    for (std::size_t index = 0; index < max_rank; ++index)
    {
        output_shape.push_back(std::max(left_shape.at(index), right_shape.at(index)));
    }

    return {output_shape, left_shape, right_shape};
}

/// \brief Calculate the output shape of numpy-style broadcast operation for all input nodes.
///
/// This function finds the maximum tensor shape that will be the result of element-wise operation
/// that will be applied to the inputs vector. The function also prepares the shape of each input
/// for the element-wise operation by left-padding those shapes so that their rank is equal to
/// the target_shape's rank.
///
/// \param inputs A vector of input nodes for which a common shape should be found
/// \return A pair that contains the target shape as its first object and a vector of padded
///         input shapes ready to be broadcasted as the second object
static std::pair<ngraph::Shape, std::vector<ngraph::Shape>>
    get_numpy_broadcast_shapes(const ngraph::NodeVector& inputs)
{
    auto shape_left_fold = [](const ngraph::Shape& accumulator,
                              const std::shared_ptr<ngraph::Node>& input) {
        // TODO: in a separate PR remove the 'get_numpy_broadcast_shape' function
        return get_numpy_broadcast_shape(accumulator, input->get_shape()).at(0);
    };

    ngraph::Shape target_shape =
        std::accumulate(std::begin(inputs), std::end(inputs), ngraph::Shape{}, shape_left_fold);

    std::vector<ngraph::Shape> full_shapes;
    for (const std::shared_ptr<ngraph::Node>& input : inputs)
    {
        ngraph::Shape padded_shape = input->get_shape();
        padded_shape.insert(std::begin(padded_shape), target_shape.size() - padded_shape.size(), 1);
        full_shapes.push_back(std::move(padded_shape));
    }

    return {target_shape, full_shapes};
}

/// \brief      Broadcast input node.
///
/// \note       The source shape does not have to be the actual shape of input node. However
///             it should be a superset of it (containing it as a continuous subset). This implies
///             we may expand the number of axes of input node.
///             The ranks of source_shape and output_shape must be equal. This means that the
///             source_shape has to be padded with ones for this operation.
///
/// \param[in]  node          The input Node to be broadcasted.
/// \param[in]  output_shape  The output shape.
/// \param[in]  source_shape  The source shape from which we want to broadcast input node.
///
/// \return     The broadcasted Node.
///
static std::shared_ptr<ngraph::Node>
    broadcast_node_numpy_style(const std::shared_ptr<ngraph::Node>& node,
                               const ngraph::Shape& output_shape,
                               const ngraph::Shape& source_shape)
{
    if (source_shape.size() != output_shape.size())
    {
        NGRAPH_WARN << "Ranks of source_shape and output_shape dont match: " << source_shape.size()
                    << " vs " << output_shape.size();
    }

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

            return {broadcast_node_numpy_style(left, output_shape, left_full_shape),
                    broadcast_node_numpy_style(right, output_shape, right_full_shape)};
        }

        NodeVector numpy_style_broadcast(NodeVector inputs)
        {
            if (inputs.size() <= 1)
            {
                return inputs;
            }

            // find the output tensor's shape, then broadcast all inputs so that they are compatible
            auto bcast_shapes = get_numpy_broadcast_shapes(inputs);

            NodeVector broadcasted_inputs;
            for (std::size_t i = 0; i < inputs.size(); ++i)
            {
                const std::shared_ptr<ngraph::Node> input_node = inputs[i];

                Shape source_shape = input_node->get_shape();
                broadcasted_inputs.push_back(broadcast_node_numpy_style(
                    inputs[i], bcast_shapes.first, bcast_shapes.second[i]));
            }

            return broadcasted_inputs;
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

            return {broadcast_node_numpy_style(left, left_output_shape, left_full_shape),
                    broadcast_node_numpy_style(right, right_output_shape, right_full_shape)};
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
            std::size_t num_ones = 0;
            for (std::size_t dimension : new_right_shape)
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
            std::vector<std::size_t> result(output_shape.size() - input_shape.size());
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
