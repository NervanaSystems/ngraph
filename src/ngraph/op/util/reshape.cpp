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

#include <numeric>

#include "ngraph/node.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/shape.hpp"

#include "reshape.hpp"

using namespace ngraph;

std::shared_ptr<Node> op::util::reshape(const std::shared_ptr<Node>& node,
                                        const AxisVector& axis_order,
                                        const Shape& shape)
{
    return std::make_shared<op::Reshape>(
        node, op::util::get_default_axis_vector(node->get_shape().size()), shape);
}

AxisVector op::util::get_default_axis_vector(std::size_t data_shape_rank, std::size_t start_value)
{
    AxisVector axes(data_shape_rank);
    std::iota(std::begin(axes), std::end(axes), start_value);
    return axes;
}

std::shared_ptr<Node> op::util::reorder_axes(const std::shared_ptr<Node>& node,
                                             std::vector<std::size_t> axes_order = {})
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
    return std::make_shared<op::Reshape>(node, axis_vector, out_shape);
}
