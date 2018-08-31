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

#include <numeric>
#include <vector>

#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/reshape.hpp"

#include "broadcasting.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        std::vector<Shape> calculate_numpy_broadcast_shape(Shape left_shape, Shape right_shape)
        {
            Shape output_shape;
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

        NodeVector binary_op_numpy_broadcasting(const std::shared_ptr<Node>& left,
                                                const std::shared_ptr<Node>& right)
        {
            auto left_shape = left->get_shape();
            auto right_shape = right->get_shape();
            auto numpy_shapes = calculate_numpy_broadcast_shape(left_shape, right_shape);
            auto output_shape = numpy_shapes.at(0);
            auto left_full_shape = numpy_shapes.at(1);
            auto right_full_shape = numpy_shapes.at(2);

            std::vector<size_t> left_ones_positions;
            std::vector<size_t> right_ones_positions;
            Shape new_left_shape;
            Shape new_right_shape;
            for (auto index = 0; index < output_shape.size(); ++index)
            {
                (left_full_shape.at(index) == 1)
                    ? left_ones_positions.push_back(index)
                    : new_left_shape.push_back(left_full_shape.at(index));
                (right_full_shape.at(index) == 1)
                    ? right_ones_positions.push_back(index)
                    : new_right_shape.push_back(right_full_shape.at(index));
            }

            // Generate an increasing sequence (0,1,2,3..) as input_order for Reshape
            std::vector<size_t> left_input_order(left->get_shape().size());
            std::iota(std::begin(left_input_order), std::end(left_input_order), 0);

            std::shared_ptr<Node> broadcasted_left =
                std::make_shared<op::Reshape>(left, left_input_order, new_left_shape);

            // Generate an increasing sequence (0,1,2,3..) as input_order for Reshape
            std::vector<size_t> right_input_order(right->get_shape().size());
            std::iota(std::begin(right_input_order), std::end(right_input_order), 0);

            std::shared_ptr<Node> broadcasted_right =
                std::make_shared<op::Reshape>(right, right_input_order, new_right_shape);

            broadcasted_left = std::make_shared<op::Broadcast>(
                broadcasted_left, output_shape, left_ones_positions);

            broadcasted_right = std::make_shared<op::Broadcast>(
                broadcasted_right, output_shape, right_ones_positions);

            return {broadcasted_left, broadcasted_right};
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
