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
        if (rotations[axis] == 0)
        {
            continue;
        }

        if (rotations[axis] >= shape[axis])
        {
            throw ngraph_error("Rotation size greater than or equal to input tensor axis size");
        }

        Coordinate left_bound = upper_bound;
        left_bound[axis] = rotations[axis];
        auto left_slice = std::make_shared<op::Slice>(concat, lower_bound, left_bound);

        Coordinate right_bound = lower_bound;
        right_bound[axis] = rotations[axis];
        auto right_slice = std::make_shared<op::Slice>(concat, right_bound, upper_bound);

        concat = std::make_shared<op::Concat>(NodeVector{right_slice, left_slice}, axis);
    }

    return concat;
}
