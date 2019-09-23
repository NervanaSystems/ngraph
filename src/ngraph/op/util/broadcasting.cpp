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
#include "ngraph/util.hpp"

/// \brief Calculate the output shape of numpy-style broadcast operation for two shapes.
///
/// more info:
/// https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#general-broadcasting-rules
/// example:
/// left: [3, 1, 10] right: [5, 1]
/// return: [3, 5, 10]
///
/// \param left_shape First input shape.
/// \param right_shape Second input Shape.
/// \return Broadcast shape of input shapes.
static ngraph::Shape calculate_broadcast_shape(ngraph::Shape left_shape, ngraph::Shape right_shape)
{
    ngraph::Shape result;
    auto left_rank = left_shape.size();
    auto right_rank = right_shape.size();
    auto max_rank = std::max(left_rank, right_rank);

    // left-pad the left_shape with zeros
    left_shape.insert(std::begin(left_shape), max_rank - left_rank, 0);
    // left-pad the right_shape with zeros
    right_shape.insert(std::begin(right_shape), max_rank - right_rank, 0);

    for (std::size_t index = 0; index < max_rank; ++index)
    {
        result.push_back(std::max(left_shape.at(index), right_shape.at(index)));
    }

    return result;
};

/// \brief Calculate the output shape of numpy-style broadcast operation for all input shapes.
///
/// This function finds the maximum tensor shape that will be the result of element-wise operation
/// that will be applied to the input shapes vector. The function also prepares the shape of each
/// input for the element-wise operation by left-padding those shapes so that their rank is equal
/// to the left_shape's rank.
///
/// \param input_shapes A vector of input shapes for which a common shape should be found
/// \return A pair that contains the target shape as its first object and a vector of padded
///         input shapes ready to be broadcasted as the second object
static std::pair<ngraph::Shape, std::vector<ngraph::Shape>>
    get_numpy_broadcast_shapes(const std::vector<ngraph::Shape>& input_shapes)
{
    ngraph::Shape target_shape = std::accumulate(std::begin(input_shapes),
                                                 std::end(input_shapes),
                                                 ngraph::Shape{},
                                                 calculate_broadcast_shape);

    std::vector<ngraph::Shape> full_shapes;
    for (const ngraph::Shape& input : input_shapes)
    {
        ngraph::Shape padded_shape{input};
        padded_shape.insert(std::begin(padded_shape), target_shape.size() - padded_shape.size(), 1);
        full_shapes.push_back(std::move(padded_shape));
    }

    return {target_shape, full_shapes};
}

/// \brief Calculate the output shape of numpy-style broadcast operation for all input nodes.
///
/// \param inputs A vector of input nodes for which a common shape should be found
/// \return A pair that contains the target shape as its first object and a vector of padded
///         input shapes ready to be broadcasted as the second object
static std::pair<ngraph::Shape, std::vector<ngraph::Shape>>
    get_numpy_broadcast_shapes(const ngraph::NodeVector& inputs)
{
    std::vector<ngraph::Shape> input_shapes;

    for (const auto& input : inputs)
    {
        input_shapes.push_back(input->get_shape());
    }

    return get_numpy_broadcast_shapes(input_shapes);
}

static std::pair<ngraph::Shape, std::vector<ngraph::Shape>>
    get_numpy_broadcast_shapes(const ngraph::OutputVector& values)
{
    std::vector<ngraph::Shape> input_shapes;

    for (const auto& input : values)
    {
        input_shapes.push_back(input.get_shape());
    }

    return get_numpy_broadcast_shapes(input_shapes);
}

/// \brief      Broadcast input node.
///
/// \note       The source shape does not have to be the actual shape of input node. However
///             it should be a superset of it (containing it as a continuous subset). This implies
///             we may expand the number of axes of input node.
///             The ranks of source_shape and output_shape must be equal. This means that the
///             source_shape has to be padded with ones for this operation.
///
/// \param[in]  value         The input Node to be broadcast.
/// \param[in]  output_shape  The output shape.
/// \param[in]  source_shape  The source shape from which we want to broadcast input node.
///
/// \return     The broadcasted Node.
///
static std::shared_ptr<ngraph::Node>
    broadcast_node_numpy_style(const ngraph::Output<ngraph::Node>& value,
                               const ngraph::Shape& output_shape,
                               const ngraph::Shape& source_shape)
{
    // If node already has the required shape, return original node
    if (output_shape == value.get_shape())
    {
        return value.as_single_output_node();
    }

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
    ngraph::Output<ngraph::Node> broadcasted_value = std::make_shared<ngraph::op::Reshape>(
        value, ngraph::get_default_order(value.get_shape()), squeezed_shape);

    return std::make_shared<ngraph::op::Broadcast>(broadcasted_value, output_shape, broadcast_axes);
}

namespace ngraph
{
    namespace op
    {
        OutputVector numpy_style_broadcast_values(const OutputVector& values)
        {
            if (values.size() <= 1)
            {
                return values;
            }

            // find the output tensor's shape, then broadcast all inputs so that they are compatible
            auto bcast_shapes = get_numpy_broadcast_shapes(values);

            OutputVector broadcasted_inputs;
            for (std::size_t i = 0; i < values.size(); ++i)
            {
                broadcasted_inputs.push_back(broadcast_node_numpy_style(
                    values[i], bcast_shapes.first, bcast_shapes.second[i]));
            }
            return broadcasted_inputs;
        }

        NodeVector numpy_style_broadcast(const NodeVector& inputs)
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
                broadcasted_inputs.push_back(broadcast_node_numpy_style(
                    inputs[i], bcast_shapes.first, bcast_shapes.second[i]));
            }
            return broadcasted_inputs;
        }

        std::shared_ptr<ngraph::Node> numpy_style_broadcast(const Output<ngraph::Node>& value,
                                                            const Shape& shape)
        {
            auto bcast_shape = get_numpy_broadcast_shapes({value.get_shape(), shape});
            return broadcast_node_numpy_style(value, bcast_shape.first, bcast_shape.second[0]);
        }

        NodeVector
            numpy_style_broadcast_for_matmul_operation(const std::shared_ptr<ngraph::Node>& left,
                                                       const std::shared_ptr<ngraph::Node>& right)
        {
            const auto& left_shape = left->get_shape();
            const auto& right_shape = right->get_shape();
            // Broadcast only _stack of matrices_ axes.
            const auto& numpy_shapes = get_numpy_broadcast_shapes(
                {Shape{std::begin(left_shape), std::next(std::end(left_shape), -2)},
                 Shape{std::begin(right_shape), std::next(std::end(right_shape), -2)}});

            // Prepare tensors output shapes with broadcasted _stack of matrices_ axes.
            auto left_output_shape = numpy_shapes.first;
            auto right_output_shape = numpy_shapes.first;
            // Append the last two axes original dimensions.
            left_output_shape.insert(std::end(left_output_shape),
                                     std::next(std::begin(left_shape), left_shape.size() - 2),
                                     std::end(left_shape));
            right_output_shape.insert(std::end(right_output_shape),
                                      std::next(std::begin(right_shape), right_shape.size() - 2),
                                      std::end(right_shape));

            auto left_full_shape = numpy_shapes.second.at(0);
            auto right_full_shape = numpy_shapes.second.at(1);
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

        OutputVector
            numpy_style_broadcast_values_for_matmul_operation(const Output<ngraph::Node>& left,
                                                              const Output<ngraph::Node>& right)
        {
            const auto& left_shape = left.get_shape();
            const auto& right_shape = right.get_shape();
            // Broadcast only _stack of matrices_ axes.
            const auto& numpy_shapes = get_numpy_broadcast_shapes(
                {Shape{std::begin(left_shape), std::next(std::end(left_shape), -2)},
                 Shape{std::begin(right_shape), std::next(std::end(right_shape), -2)}});

            // Prepare tensors output shapes with broadcasted _stack of matrices_ axes.
            auto left_output_shape = numpy_shapes.first;
            auto right_output_shape = numpy_shapes.first;
            // Append the last two axes original dimensions.
            left_output_shape.insert(std::end(left_output_shape),
                                     std::next(std::begin(left_shape), left_shape.size() - 2),
                                     std::end(left_shape));
            right_output_shape.insert(std::end(right_output_shape),
                                      std::next(std::begin(right_shape), right_shape.size() - 2),
                                      std::end(right_shape));

            auto left_full_shape = numpy_shapes.second.at(0);
            auto right_full_shape = numpy_shapes.second.at(1);
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
                right, ngraph::get_default_order(right_shape), new_right_shape);

            // Move broadcast start axis parameter to right
            start_match_axis += num_ones;

            auto broadcast_right = std::make_shared<ngraph::op::Broadcast>(
                reshape_right,
                left_shape,
                calculate_broadcast_axes(left_shape, new_right_shape, start_match_axis));

            return {left, broadcast_right};
        }

        OutputVector
            legacy_style_broadcast_values_for_binary_operation(const Output<ngraph::Node>& left,
                                                               const Output<ngraph::Node>& right,
                                                               size_t start_match_axis)
        {
            const auto& left_shape = left.get_shape();
            const auto& right_shape = right.get_shape();

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
                right, ngraph::get_default_order(right_shape), new_right_shape);

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
            // output_shape_size, excluding values in range:
            // [start_match_axis, start_match_axis + input_shape.size()]
            std::iota(std::begin(result), std::begin(result) + start_match_axis, 0);
            std::iota(std::begin(result) + start_match_axis,
                      std::end(result),
                      start_match_axis + input_shape.size());
            return result;
        }

    } // namespace  op

} // namespace  ngraph
