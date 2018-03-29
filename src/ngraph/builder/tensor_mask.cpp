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

#include <numeric>

#include "ngraph/builder/tensor_mask.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/less.hpp"
#include "ngraph/op/reshape.hpp"

using namespace ngraph;

std::shared_ptr<Node> ngraph::builder::tensor_mask(const std::shared_ptr<Node>& sequence_lengths,
                                                   size_t sequence_axis,
                                                   size_t batch_axis,
                                                   Shape mask_shape)
{
    if (sequence_axis >= mask_shape.size())
    {
        throw ngraph_error("Sequence axis must be in range 0..mask_shape rank");
    }

    if (batch_axis >= mask_shape.size())
    {
        throw ngraph_error("Sequence axis must be in range 0..mask_shape rank");
    }

    // all axes except the sequence axis
    AxisSet non_sequence_axes;
    // all axes except the batch axis
    AxisSet non_batch_axes;

    for (auto axis = 0; axis < mask_shape.size(); ++axis)
    {
        if (axis != sequence_axis)
        {
            non_sequence_axes.insert(axis);
        }
        if (axis != batch_axis)
        {
            non_batch_axes.insert(axis);
        }
    }

    // broadcast sequence lengths to mask shape along all non-batch axes
    auto broadcast_sequence_lengths =
        std::make_shared<op::Broadcast>(sequence_lengths, mask_shape, non_batch_axes);

    // create sequence data [0, ..., max_sequence_length]
    auto max_sequence_length = mask_shape[sequence_axis];
    std::vector<uint32_t> sequence_data(max_sequence_length);
    std::iota(sequence_data.begin(), sequence_data.end(), 0);

    // create sequence constant
    auto sequence =
        std::make_shared<op::Constant>(element::u32, Shape{max_sequence_length}, sequence_data);

    // convert sequence to input type
    auto convert_sequence =
        std::make_shared<op::Convert>(sequence, sequence_lengths->get_element_type());

    // broadcast sequence to mask shape along all non-sequence axes
    auto broadcast_sequence =
        std::make_shared<op::Broadcast>(convert_sequence, mask_shape, non_sequence_axes);

    // mask = sequence_length < sequence
    return std::make_shared<op::Less>(broadcast_sequence, broadcast_sequence_lengths);
}
