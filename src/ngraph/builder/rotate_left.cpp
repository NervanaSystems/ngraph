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

#include "ngraph/builder/rotate_left.hpp"
#include "ngraph/coordinate.hpp"
#include "ngraph/node_vector.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/slice.hpp"

using namespace ngraph;

std::shared_ptr<Node> ngraph::builder::rotate_left(const std::shared_ptr<Node>& node,
                                                   AxisVector rotations)
{
    auto shape = node->get_shape();
    if (shape.size() != rotations.size())
    {
        throw ngraph_error("Rotation rank much match input tensor rank");
    }

    auto concat = node;
    Coordinate lower_bound(shape.size(), 0);
    Coordinate upper_bound = shape;

    for (size_t axis = 0; axis < rotations.size(); ++axis)
    {
        auto rotation_size = rotations[axis];
        if (rotation_size == 0)
        {
            // nothing to do
            continue;
        }

        auto axis_size = shape[axis];
        while (rotation_size >= axis_size)
        {
            // normalize rotatation to axis size
            rotation_size -= axis_size;
        }

        // get left_slice = [0:rotation_size)
        Coordinate left_upper_bound = upper_bound;
        left_upper_bound[axis] = rotation_size;
        auto left_slice = std::make_shared<op::Slice>(concat, lower_bound, left_upper_bound);

        // get right_slice = [rotation_size:axis_size)
        Coordinate right_lower_bound = lower_bound;
        right_lower_bound[axis] = rotation_size;
        auto right_slice = std::make_shared<op::Slice>(concat, right_lower_bound, upper_bound);

        // concatenate right_slice:left_slice
        concat = std::make_shared<op::Concat>(NodeVector{right_slice, left_slice}, axis);
    }

    return concat;
}
