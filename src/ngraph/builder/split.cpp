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

#include "ngraph/builder/split.hpp"
#include "ngraph/op/slice.hpp"

using namespace ngraph;

namespace
{
    inline size_t get_valid_array_index(size_t idx, size_t axis_size)
    {
        return std::min(idx, axis_size);
    }

    std::shared_ptr<op::Slice> make_ng_slice(const Output<Node>& output,
                                             const std::vector<size_t>& axes,
                                             const std::vector<size_t>& starts,
                                             const std::vector<size_t>& ends)
    {
        std::vector<size_t> upper_bounds{output.get_shape()};
        std::vector<size_t> lower_bounds(upper_bounds.size());
        for (size_t index{0}; index < axes.size(); ++index)
        {
            size_t axis{axes.at(index)};
            lower_bounds.at(axis) =
                get_valid_array_index(starts.at(index), output.get_shape().at(axis));
            upper_bounds.at(axis) =
                get_valid_array_index(ends.at(index), output.get_shape().at(axis));
        }
        return std::make_shared<op::Slice>(output, lower_bounds, upper_bounds);
    }
}

NodeVector builder::split(const Output<ngraph::Node>& value,
                          const std::vector<size_t>& length_parts,
                          size_t axis)
{
    size_t start_index{0};
    NodeVector outputs;
    for (const auto& length_part : length_parts)
    {
        size_t end_index{start_index + length_part};
        outputs.push_back(make_ng_slice(value, {axis}, {start_index}, {end_index}));
        start_index = end_index;
    }
    return outputs;
}

NodeVector builder::split(const Output<Node>& value, size_t split_parts, int axis)
{
    size_t axis_to_split{static_cast<size_t>(axis)};
    if (axis < 0)
    {
        axis_to_split = value.get_shape().size() + axis;
    }

    size_t length_axis_to_split{value.get_shape().at(axis_to_split)};
    std::vector<size_t> length_parts(split_parts, length_axis_to_split / split_parts);
    return split(value, length_parts, axis_to_split);
}
