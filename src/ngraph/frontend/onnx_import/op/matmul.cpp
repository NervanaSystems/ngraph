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

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <vector>

#include "ngraph/assertion.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/reshape.hpp"

#include "exceptions.hpp"
#include "matmul.hpp"
#include "utils/broadcasting.hpp"
#include "utils/common.hpp"
#include "utils/reshape.hpp"

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
                    NodeVector ng_inputs{node.get_ng_inputs()};
                    auto left{ng_inputs.at(0)};
                    auto right{ng_inputs.at(1)};
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
                    NodeVector broadcasted_nodes =
                        numpy_style_broadcast_for_matmul_operation(left, right);

                    left = broadcasted_nodes.at(0);
                    right = broadcasted_nodes.at(1);
                    auto left_shape = left->get_shape();
                    auto right_shape = right->get_shape();

                    // Reorder axes to prepare data for Ngraph Dot op.
                    // Move second from end axis to the begining.
                    std::vector<std::size_t> axes_order{right_shape.size() - 2};
                    auto tmp_range = common::get_monotonic_range(right_shape.size() - 2);
                    axes_order.insert(
                        std::end(axes_order), std::begin(tmp_range), std::end(tmp_range));
                    axes_order.push_back(right_shape.size() - 1);
                    right = reshape::reorder_axes(right, axes_order);

                    // TODO: remove, just check wheter shapes are equal
                    bool equal_shapes = std::equal(std::begin(left_shape),
                                                   std::next(std::begin(left_shape), left_shape.size() - 1),
                                                   std::next(std::begin(right->get_shape())));
                    NGRAPH_ASSERT(equal_shapes) << "Arguments have unequal not reduced axes dimensions";

                    // Perform multiply operation.
                    auto dot_node = std::make_shared<ngraph::op::Dot>(left, right);

                    // Slice data to get expected result.
                    // 3D case
                    Shape dot_shape = dot_node->get_shape();
                    Shape lower_bounds(dot_shape.size());
                    Shape upper_bounds = dot_shape;

                    NodeVector result_slices;
                    for (auto i = 0; i < dot_shape.at(0); ++i)
                    {
                        // Coupled axes
                        lower_bounds.at(0) = i;
                        upper_bounds.at(0) = i + 1;
                        lower_bounds.at(left_shape.size() - 1) = i;
                        upper_bounds.at(left_shape.size() - 1) = i + 1;

                        auto sliced_dot = std::make_shared<ngraph::op::Slice>(
                            dot_node, lower_bounds, upper_bounds);
                        Shape sliced_shape{1, left_shape.at(1), right_shape.back()};
                        result_slices.push_back(std::make_shared<ngraph::op::Reshape>(
                            sliced_dot,
                            reshape::get_default_axis_vector(sliced_dot->get_shape().size()),
                            sliced_shape));
                    }

                    return {std::make_shared<ngraph::op::Concat>(result_slices, 0)};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
