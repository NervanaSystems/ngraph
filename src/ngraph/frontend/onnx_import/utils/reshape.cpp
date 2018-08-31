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
#include <functional>
#include <numeric>
#include <stdexcept>

#include "ngraph/axis_vector.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/shape.hpp"

#include "exceptions.hpp"
#include "utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace reshape
        {
            std::shared_ptr<ngraph::Node> flatten(const std::shared_ptr<ngraph::Node>& node,
                                                  int axis)
            {
                auto data_shape = node->get_shape();

                size_t first_dim_size = 1;
                size_t last_dim_size = 1;

                //  First dimension of output tensor is the product of [d_0, ... d_{axis-1}] dimensions of input tensor.
                //  The last dimension is the product of the rest of input tensor dimensions: [d_{axis}, ..., d_n]
                for (auto index = 0; index < data_shape.size(); ++index)
                {
                    last_dim_size *= data_shape.at(index);
                    if (index < axis)
                    {
                        first_dim_size = last_dim_size;
                    }
                }

                last_dim_size /= first_dim_size;

                // Generate an increasing sequence (0,1,2,3..) as input_order for Reshape
                std::vector<size_t> input_order(data_shape.size());
                std::iota(std::begin(input_order), std::end(input_order), 0);

                return std::make_shared<ngraph::op::Reshape>(
                    node, AxisVector{input_order}, Shape{first_dim_size, last_dim_size});
            }

            AxisVector get_default_axis_vector(std::size_t data_shape_size, std::size_t start_value)
            {
                AxisVector axis_vector(data_shape_size);
                std::iota(std::begin(axis_vector), std::end(axis_vector), start_value);
                return axis_vector;
            }

            std::vector<std::size_t> infer_dimensions(const std::string& node_name,
                                                      const std::vector<std::size_t>& input_shape,
                                                      const std::vector<std::size_t>& output_shape)
            {
                // If an output dimension is equal to zero its actual value is copied from the input
                // shape argument.
                for (std::size_t idx = 0; idx < output_shape.size(); ++idx)
                {
                    if (output_shape.at(idx) == 0)
                    {
                        if (idx < input_shape.size())
                        {
                            output_shape.at(idx) = input_shape.at(idx);
                        }
                        else
                        {
                            throw error::parameter::Value(
                                "Reshape",
                                node_name,
                                "can not copy dimension from the input data shape since requested "
                                "index is out of range.");
                        }
                    }
                }

                // Check whether there are dimensions equal to -1 in output_shape. There may be at most
                // one such case. Its value is then inferred from the size of the tensor and the
                // remaining dimensions.
                auto neg_value_it = std::find(std::begin(output_shape), std::end(output_shape), -1);
                auto sec_neg_value_it = std::find(neg_value_it, std::end(output_shape), -1);
                if (sec_neg_value_it != std::end(output_shape))
                {
                    throw error::parameter::Value("Reshape",
                                                  node_name,
                                                  "more than one dimension is set to (-1). Only "
                                                  "one dimension value can be inferred.");
                }
                if (neg_value_it != std::end(output_shape))
                {
                    *neg_value_it = 1;
                    std::size_t input_shape_product =
                        std::accumulate(std::begin(input_shape),
                                        std::end(input_shape),
                                        1UL,
                                        std::multiplies<std::size_t>());
                    std::size_t output_shape_product =
                        std::accumulate(std::begin(output_shape),
                                        std::end(output_shape),
                                        1UL,
                                        std::multiplies<std::size_t>());
                    *neg_value_it = input_shape_product / output_shape_product;
                }

                return output_shape;
            }

            std::shared_ptr<ngraph::Node> reorder_axes(const std::shared_ptr<ngraph::Node>& node,
                                                       std::vector<size_t> axes_order = {})
            {
                Shape out_shape = node->get_shape();
                if (axes_order.empty())
                {
                    axes_order.resize(out_shape.size());
                    std::iota(std::begin(axes_order), std::end(axes_order), 0);
                }
                else
                {
                    for (std::size_t i = 0; i < axes_order.size(); ++i)
                    {
                        out_shape[i] = node->get_shape().at(axes_order.at(i));
                    }
                }

                auto axis_vector = AxisVector{std::begin(axes_order), std::end(axes_order)};
                return std::make_shared<ngraph::op::Reshape>(node, axis_vector, out_shape);
            }

            std::shared_ptr<ngraph::Node> transpose(const std::shared_ptr<ngraph::Node>& node)
            {
                std::vector<size_t> axes_order(node->get_shape().size());
                std::iota(std::begin(axes_order), std::end(axes_order), 0);
                std::reverse(std::begin(axes_order), std::end(axes_order));
                return reorder_axes(node, axes_order);
            }

        } // namespace  reshape
    }     // namespace onnx_import
} // namespace ngraph
