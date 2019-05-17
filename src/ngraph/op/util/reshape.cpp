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
#include <util.hpp>

#include "ngraph/node.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/util.hpp"

#include "reshape.hpp"

using namespace ngraph;
using namespace std;

namespace
{
    inline std::size_t get_valid_array_index(std::size_t idx, std::size_t axis_size)
    {
        return std::min(idx, axis_size);
    }

    std::shared_ptr<op::Slice> make_ng_slice(const std::shared_ptr<ngraph::Node>& node,
                                             const std::vector<std::size_t>& axes,
                                             const std::vector<std::size_t>& starts,
                                             const std::vector<std::size_t>& ends)
    {
        std::vector<std::size_t> upper_bounds{node->get_shape()};
        std::vector<std::size_t> lower_bounds(upper_bounds.size());
        for (std::size_t index{0}; index < axes.size(); ++index)
        {
            std::size_t axis{axes.at(index)};
            lower_bounds.at(axis) =
                    get_valid_array_index(starts.at(index), node->get_shape().at(axis));
            upper_bounds.at(axis) =
                    get_valid_array_index(ends.at(index), node->get_shape().at(axis));
        }
        return std::make_shared<op::Slice>(node, lower_bounds, upper_bounds);
    }
}

shared_ptr<Node> op::util::reshape(const shared_ptr<Node>& node, const Shape& shape)
{
    return make_shared<op::Reshape>(node, get_default_order(node->get_shape().size()), shape);
}

shared_ptr<Node> op::util::reorder_axes(const shared_ptr<Node>& node,
                                        vector<size_t> axes_order = {})
{
    Shape out_shape = node->get_shape();
    if (axes_order.empty())
    {
        axes_order.resize(out_shape.size());
        iota(begin(axes_order), end(axes_order), 0);
    }
    else
    {
        for (size_t i = 0; i < axes_order.size(); ++i)
        {
            out_shape[i] = node->get_shape().at(axes_order.at(i));
        }
    }

    auto axis_vector = AxisVector{begin(axes_order), end(axes_order)};
    return make_shared<op::Reshape>(node, axis_vector, out_shape);
}

shared_ptr<Node> op::util::transpose(const shared_ptr<Node>& node)
{
    vector<size_t> axes_order(node->get_shape().size());
    iota(begin(axes_order), end(axes_order), 0);
    reverse(begin(axes_order), end(axes_order));
    return op::util::reorder_axes(node, axes_order);
}

shared_ptr<Node> op::util::flatten(const shared_ptr<Node>& node, int axis)
{
    auto data_shape = node->get_shape();

    //  First dimension of output tensor is the product of [d_0, ... d_{axis-1}] dimensions of input tensor.
    //  The last dimension is the product of the rest of input tensor dimensions: [d_{axis}, ..., d_n]
    size_t first_dim_size =
        accumulate(begin(data_shape), next(begin(data_shape), axis), 1UL, multiplies<size_t>());

    size_t last_dim_size =
        accumulate(next(begin(data_shape), axis), end(data_shape), 1UL, multiplies<size_t>());

    return make_shared<op::Reshape>(
        node, get_default_order(data_shape.size()), Shape{first_dim_size, last_dim_size});
}

NodeVector op::util::split(const shared_ptr<ngraph::Node>& node,
                 const vector<size_t>& length_parts,
                 size_t axis)
{
    size_t start_index{0};
    NodeVector outputs;
    for (const auto& length_part : length_parts)
    {
        size_t end_index{start_index + length_part};
        outputs.push_back(make_ng_slice(node, {axis}, {start_index}, {end_index}));
        start_index = end_index;
    }
    return outputs;
}

NodeVector op::util::split(const shared_ptr<ngraph::Node>& node, size_t split_parts, int axis)
{
    size_t axis_to_split{static_cast<size_t>(axis)};
    if (axis < 0)
    {
        axis_to_split = node->get_shape().size() + axis;
    }

    size_t length_axis_to_split{node->get_shape().at(axis_to_split)};
    vector<size_t> length_parts(split_parts,
                                          length_axis_to_split / split_parts);
    return split(node, length_parts, axis_to_split);
}
