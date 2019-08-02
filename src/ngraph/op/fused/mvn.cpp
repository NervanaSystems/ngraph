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

#include "mvn.hpp"
#include "ngraph/builder/reduce_ops.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/util/broadcasting.hpp"

using namespace std;
using namespace ngraph;

const string op::MVN::type_name{"MVN"};

op::MVN::MVN(const std::shared_ptr<Node>& data,
             bool across_channels,
             bool normalize_variance,
             double eps)
    : FusedOp(check_single_output_args({data}))
    , m_eps{eps}
    , m_across_channels{across_channels}
    , m_normalize_variance{normalize_variance}
{
    constructor_validate_and_infer_types();

    // if m_across_channels is true we should calculate mean and variance per batch
    // else we calculate these per channel
    m_reduction_axes.insert(0);
    size_t start_axis = m_across_channels ? 1 : 2;
    for (size_t i = start_axis; i < data->get_shape().size(); ++i)
    {
        m_reduction_axes.insert(i);
    }
}

op::MVN::MVN(const std::shared_ptr<Node>& data,
             AxisSet reduction_axes,
             bool normalize_variance,
             double eps)
    : FusedOp(check_single_output_args({data}))
    , m_eps{eps}
    , m_across_channels{false}
    , m_normalize_variance{normalize_variance}
    , m_reduction_axes{reduction_axes}
{
    constructor_validate_and_infer_types();
}

NodeVector op::MVN::decompose_op() const
{
    auto data = get_argument(0);
    auto data_shape = data->get_shape(); // assume that data has n and c channels.

    // calculate mean normalization
    auto mean = builder::mean(data, m_reduction_axes);
    mean = std::make_shared<op::Broadcast>(mean, data_shape, m_reduction_axes);
    auto mean_normalization = data - mean;

    if (!m_normalize_variance)
    {
        return {mean_normalization};
    }
    else
    {
        // calculate variance
        auto variance = builder::variance(data, m_reduction_axes);
        variance = make_shared<op::Sqrt>(variance);
        // add epsilon
        auto eps_node = op::Constant::create(
            data->get_element_type(), variance->get_shape(), vector<double>{m_eps});
        variance = variance + eps_node;
        variance = std::make_shared<op::Broadcast>(variance, data_shape, m_reduction_axes);

        return {mean_normalization / variance};
    }
}

shared_ptr<Node> op::MVN::copy_with_new_args(const NodeVector& new_args) const
{
    NODE_VALIDATION_CHECK(this,
                          new_args.size() == 1,
                          "Expected 1 element in new_args for the MVN op but got ",
                          new_args.size());
    return make_shared<MVN>(new_args.at(0), m_reduction_axes, m_normalize_variance, m_eps);
}
