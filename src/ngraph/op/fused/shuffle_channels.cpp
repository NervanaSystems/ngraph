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

op::ShuffleChannels::ShuffleChannels(const shared_ptr<Node>& data, const int axis, const int groups)
    : FusedOp("ShuffleChannels", {data})
    , m_axis{axis}
    , m_groups{groups}
{
    constructor_validate_and_infer_types();
}

void op::ShuffleChannels::pre_validate_and_infer_types()
{
    NODE_VALIDATION_CHECK(
        this, m_axis >= 0, "Accepted values of the 'axis' parameter for ShuffleChannels are positive integers and zero.");

    NODE_VALIDATION_CHECK(
        this, m_groups > 0, "The 'groups' parameter for ShuffleChannels needs to be greater than zero.");

    const auto data = get_argument(0);
    const auto shape = data->get_shape();
    const auto channel_dim_size = shape.at(m_axis);

    NODE_VALIDATION_CHECK(
        this, m_axis < shape.size(), "The 'axis' parameter for ShuffleChannels needs to be less than the input tensor rank.");

    NODE_VALIDATION_CHECK(
        this, shape.size() == 4, "The input tensor's shape is expected to be 4D.");

    NODE_VALIDATION_CHECK(
        this,
        channel_dim_size % m_groups == 0,
        "The channel dimension size has to be divisible by the 'groups' parameter value");
}

NodeVector op::ShuffleChannels::decompose_op() const
{
    const auto data = get_argument(0);
    const auto data_shape = data->get_shape();

    size_t N = data_shape.at(0);
    size_t C = data_shape.at(1);
    size_t H = data_shape.at(2);
    size_t W = data_shape.at(3);

    // if the axis parameter is different than one, the C value must be taken from dimension 0
    if (m_axis != 1)
    {
        std::swap(N, C);
    }

    const auto reshaped = util::reshape(data, {N, static_cast<size_t>(m_groups), C / m_groups, H * W});
    const auto reordered = util::reorder_axes(reshaped, {0, 2, 1, 3});
    const auto shuffled = util::reshape(data, {N, C, H, W});

    return {shuffled};
}

shared_ptr<Node> op::ShuffleChannels::copy_with_new_args(const NodeVector& new_args) const
{
    NODE_VALIDATION_CHECK(this,
                          new_args.size() == 1,
                          "Expected 1 element in new_args for the ShuffleChannels op but got ",
                          new_args.size());

    return make_shared<ShuffleChannels>(new_args.at(0), m_axis, m_groups);
}
