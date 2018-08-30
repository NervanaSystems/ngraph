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

#include "ngraph/op/reshape.hpp"

#include "utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace utils
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
                    node,
                    ngraph::AxisVector{input_order},
                    ngraph::Shape{first_dim_size, last_dim_size});
            }
        } // namespace utils

        std::shared_ptr<ngraph::Node> reorder_axes(const std::shared_ptr<ngraph::Node>& node,
                                                   std::vector<size_t> axes_order = {})
        {
            ngraph::Shape out_shape = node->get_shape();
            if (axes_order.empty())
            {
                axes_order.resize(out_shape.size());
                std::iota(std::begin(axes_order), std::end(axes_order), 0);
            }
            else
            {
                for (auto i = 0; i < axes_order.size(); ++i)
                {
                    out_shape[i] = node->get_shape().at(axes_order.at(i));
                }
            }

            auto axis_vector = ngraph::AxisVector{axes_order.begin(), axes_order.end()};
            return std::make_shared<ngraph::op::Reshape>(node, axis_vector, out_shape);
        }

        std::shared_ptr<ngraph::Node> transpose(const std::shared_ptr<ngraph::Node>& node)
        {
            std::vector<size_t> axes_order(node->get_shape().size());
            std::iota(std::begin(axes_order), std::end(axes_order), 0);
            std::reverse(std::begin(axes_order), std::end(axes_order));
            return reorder_axes(node, axes_order);
        }
    } // namespace onnx_import

} // namespace ngraph
