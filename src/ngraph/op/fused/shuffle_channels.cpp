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
#include "ngraph/op/fused/shuffle_channels.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/util/reshape.hpp"

using namespace std;
using namespace ngraph;

op::ShuffleChannels::ShuffleChannels(const shared_ptr<Node>& data,
                                     const int axis,
                                     const size_t groups)
    : FusedOp("ShuffleChannels", {data})
    , m_groups{groups}
{
    if (axis < 0)
    {
        m_axis = axis + data->get_shape().size();
    }
    else
    {
        m_axis = axis;
    }

    constructor_validate_and_infer_types();
}

void op::ShuffleChannels::pre_validate_and_infer_types()
{
    const auto shape = get_argument(0)->get_shape();

    NODE_VALIDATION_CHECK(
        this, shape.size() >= 1, "The input tensor's shape is expected to be at least 1D.");

    NODE_VALIDATION_CHECK(this,
                          m_axis >= 0 && m_axis < shape.size(),
                          "The 'axis' parameter for ShuffleChannels has to point to one of the "
                          "input tensor's shape dimensions.");

    const auto channel_dim_size = shape.at(m_axis);
    NODE_VALIDATION_CHECK(
        this,
        channel_dim_size % m_groups == 0,
        "The channel dimension size has to be a multiple of the groups parameter value.");
}

NodeVector op::ShuffleChannels::decompose_op() const
{
    const auto data = get_argument(0);
    const auto& data_shape = data->get_shape();

    const auto reshaped = util::reshape(data, get_pre_shuffle_shape(data_shape));
    const auto shuffled = util::reorder_axes(reshaped, {0, 2, 1, 3});

    return {util::reshape(shuffled, data_shape)};
}

shared_ptr<Node> op::ShuffleChannels::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Expected 1 element in new_args for the ShuffleChannels op but got " +
                           std::to_string(new_args.size()));
    }

    return make_shared<ShuffleChannels>(new_args.at(0), m_axis, m_groups);
}

Shape op::ShuffleChannels::get_pre_shuffle_shape(const Shape& data_shape) const
{
    const Shape& ds = data_shape;

    // in general the resulting shape should contain the following values:
    // [0]: ds[0] * ds[1] * ... * ds[m_axis-1] (or 1 if m_axis == 0)
    // [1]: m_groups
    // [2]: ds[axis] / m_groups
    // [3]: ds[axis+1] * ds[axis+2] * ... * ds[ds.size()-1] (or 1 if m_axis points to the last elem of ds)
    Shape res(4, 1);

    for (size_t i = 0; i < m_axis; ++i)
    {
        res[0] *= ds[i];
    }

    res[1] = m_groups;
    res[2] = ds[m_axis] / m_groups;

    for (size_t i = m_axis + 1; i < ds.size(); ++i)
    {
        res[3] *= ds[i];
    }

    return res;
}
