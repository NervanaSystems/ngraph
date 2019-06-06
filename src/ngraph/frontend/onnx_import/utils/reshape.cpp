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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <iterator>
#include <numeric>
#include <vector>

#include "exceptions.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/slice.hpp"
#include "utils/common.hpp"
#include "utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace reshape
        {
            std::vector<std::size_t> infer_dimensions(const std::string& node_name,
                                                      const std::vector<std::size_t>& input_shape,
                                                      const std::vector<std::size_t>& output_shape)
            {
                std::vector<std::size_t> inferred_dims{output_shape};

                // If an output dimension is equal to zero its actual value is copied from the input
                // shape argument.
                for (std::size_t idx = 0; idx < inferred_dims.size(); ++idx)
                {
                    if (inferred_dims.at(idx) == 0)
                    {
                        NGRAPH_CHECK(idx < input_shape.size(),
                                     "Node ",
                                     node_name,
                                     " cannot copy dimension from the input data shape because "
                                     "requested index is out of range.");

                        inferred_dims.at(idx) = input_shape.at(idx);
                    }
                }

                // Check whether there are dimensions equal to -1 in output_shape. There may be at most
                // one such case. Its value is then inferred from the size of the tensor and the
                // remaining dimensions.
                auto neg_value_it =
                    std::find(std::begin(inferred_dims), std::end(inferred_dims), -1);
                if (neg_value_it != std::end(inferred_dims))
                {
                    // only single '-1' value is allowed
                    NGRAPH_CHECK(std::find(std::next(neg_value_it), std::end(inferred_dims), -1) ==
                                     std::end(inferred_dims),
                                 "Node ",
                                 node_name,
                                 " more than one dimension is set to (-1). ",
                                 "Only one dimension value can be inferred.");

                    // Set dimension value to 1 temporarily to be able to calculate its value.
                    *neg_value_it = 1;
                    std::size_t input_shape_product =
                        std::accumulate(std::begin(input_shape),
                                        std::end(input_shape),
                                        1UL,
                                        std::multiplies<std::size_t>());
                    std::size_t output_shape_product =
                        std::accumulate(std::begin(inferred_dims),
                                        std::end(inferred_dims),
                                        1UL,
                                        std::multiplies<std::size_t>());
                    *neg_value_it = input_shape_product / output_shape_product;
                }

                return inferred_dims;
            }

            std::shared_ptr<ngraph::Node> squeeze(const std::shared_ptr<ngraph::Node>& node,
                                                  std::vector<std::size_t> axes)
            {
                if (axes.empty())
                {
                    return node;
                }

                Shape in_shape{node->get_shape()};
                for (std::size_t idx = 0; idx < axes.size(); ++idx)
                {
                    in_shape.at(idx) = 0;
                }
                Shape output_shape;
                for (auto axis : in_shape)
                {
                    if (axis != 0)
                    {
                        output_shape.push_back(axis);
                    }
                }
                return ngraph::builder::reshape(node, output_shape);
            }

            std::shared_ptr<ngraph::Node> collapse(const std::shared_ptr<ngraph::Node>& node,
                                                   const std::size_t start_axis,
                                                   const std::size_t end_axis)
            {
                auto shape = node->get_shape();
                std::size_t collapsed_axis_size =
                    std::accumulate(std::next(std::begin(shape), start_axis),
                                    std::next(std::begin(shape), end_axis + 1),
                                    1UL,
                                    std::multiplies<std::size_t>());

                Shape output_shape{collapsed_axis_size};
                output_shape.insert(std::end(output_shape),
                                    std::next(std::begin(shape), end_axis + 1),
                                    std::end(shape));
                return ngraph::builder::reshape(node, output_shape);
            }

            std::shared_ptr<ngraph::Node> expand_dims(const std::shared_ptr<ngraph::Node>& node,
                                                      std::size_t axis)
            {
                Shape output_shape(node->get_shape());
                // Add empty axis at specified position.
                auto empty_axis_it = std::begin(output_shape);
                std::advance(empty_axis_it, axis);
                output_shape.insert(empty_axis_it, 1);
                return std::make_shared<ngraph::op::Reshape>(
                    node, ngraph::get_default_order(node->get_shape().size()), output_shape);
            }

            std::shared_ptr<ngraph::Node>
                interpret_as_scalar(const std::shared_ptr<ngraph::Node>& node)
            {
                Shape node_shape = node->get_shape();

                // If node is already a scalar, return original
                if (node_shape.empty())
                {
                    return node;
                }

                NGRAPH_CHECK((shape_size(node_shape) == 1),
                             "Scalar value can't be derived from a node with ",
                             node_shape);

                return ngraph::builder::reshape(node, Shape{});
            }

        } // namespace  reshape
    }     // namespace onnx_import
} // namespace ngraph
