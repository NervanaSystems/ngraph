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
#include <functional>
#include <iterator>
#include <numeric>

#include "ngraph/axis_vector.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

shared_ptr<Node> builder::reshape(const shared_ptr<Node>& node, const Shape& shape)
{
    return make_shared<op::Reshape>(node, get_default_order(node->get_shape().size()), shape);
}

shared_ptr<Node> builder::reorder_axes(const shared_ptr<Node>& node, vector<size_t> axes_order = {})
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

shared_ptr<Node> builder::transpose(const shared_ptr<Node>& node)
{
    vector<size_t> axes_order(node->get_shape().size());
    iota(begin(axes_order), end(axes_order), 0);
    reverse(begin(axes_order), end(axes_order));
    return builder::reorder_axes(node, axes_order);
}

shared_ptr<Node> builder::flatten(const shared_ptr<Node>& node, int axis)
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
