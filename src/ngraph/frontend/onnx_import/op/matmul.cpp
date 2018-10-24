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

#pragma once

#include <cstddef>
#include <iterator>
#include <memory>
#include <vector>

#include "ngraph/op/dot.hpp"

#include "exceptions.hpp"
#include "matmul.hpp"
#include "utils/broadcasting.hpp"
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
                    auto left = ng_inputs.at(0);
                    auto right = ng_inputs.at(1);

                    // Check whether input args has compatible shapes to multiply them.
                    ASSERT_VALID_ARGUMENT(node, has_matmul_compatible_shapes(left, right))
                        << "input arg0: " << left->get_shape()
                        << " and input arg1: " right->get_shape()
                        << " data shapes are incompatible to multiply with each other.";

                    // Broadcast input arguments.
                    // FIXME: powinienem broadcast'ować tylko "stack of matrices" !!!
                    NodeVector broadcasted_nodes = numpy_style_broadcast_for_binary_operation(left, right);

                    left = broadcasted_nodes.at(0);
                    right = broadcasted_nodes.at(1);
                    auto left_shape = left->get_shape();
                    auto right_shape = right->get_shape();

                    // Reorder axes to prepare data for Ngraph Dot op.
                    if (right->get_shape().size() >= 3)
                    {
                        // Let's
                        std::vector<std::size_t> axes_order;
                        axes_order.insert(std::begin(axes_order), right_shape.rend(), right_shape.rend() + 2);
                        axes_order.insert(std::end(axes_order), std::begin(right_shape),
                            right_shape.rend() + 2);
                        right = reshape::reorder_axes(right, axes_order);
                    }

                    // Perform multiply operation.

                    // Reorder axes to get back expected result data shape.
                    if (left->get_shape().size() >= 3 || right->get_shape().size() >= 3)
                    {

                    }

                    return {std::make_shared<ngraph::op::Dot>(ng_inputs.at(0), ng_inputs.at(1))0};
                }

            } // namespace set_1

        } //namespace op

    } // namespace onnx_import

} // namespace ngraph
