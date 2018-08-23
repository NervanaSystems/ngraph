/*******************************************************************************
 * Copyright 2017-2018 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include "ngraph/op/reshape.hpp"
#include "ngraph/node_vector.hpp"

#include "core/node.hpp"
#include "core/tensor.hpp"
#include "utils/reshape.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        std::shared_ptr<ngraph::Node>
            reorder_axes(std::shared_ptr<ngraph::Node>& node,
                         std::vector<size_t> axes_order = std::vector<size_t>())
        {
            ngraph::Shape out_shape = node->get_shape();
            if (axes_order.size() == 0)
                for (int i = 0; i < out_shape.size(); ++i)
                    axes_order.push_back(i);
            else
                for (int i = 0; i < axes_order.size(); ++i)
                    out_shape[i] = node->get_shape()[axes_order[i]];

            auto axis_vector = ngraph::AxisVector(axes_order.begin(), axes_order.end());
            return std::make_shared<ngraph::op::Reshape>(node, axis_vector, out_shape);
        }

        std::shared_ptr<ngraph::Node> transpose(std::shared_ptr<ngraph::Node>& node)
        {
            std::vector<size_t> axes_order;
            for (int i = node->get_shape().size(); i > 0; --i)
                axes_order.push_back(i - 1);
            return reorder_axes(node, axes_order);
        }
    } // namespace onnx_import

} // namespace ngraph
