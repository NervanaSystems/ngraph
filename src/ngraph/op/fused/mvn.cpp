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
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/fused/mvn.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/util/broadcasting.hpp"

using namespace std;
using namespace ngraph;

op::MVN::MVN(const std::shared_ptr<Node>& data,
             bool across_channels,
             bool normalize_variance,
             double eps)
    : FusedOp("MVN", {data})
    , m_across_channels{across_channels}
    , m_normalize_variance{normalize_variance}
    , m_eps{eps}
{
    constructor_validate_and_infer_types();
}

NodeVector op::MVN::decompose_op() const
{
    auto data = get_argument(0);
    auto data_shape = data->get_shape();
    auto element_count = shape_size(data_shape);

    AxisSet reduction_axes;
    for (size_t i = 0; i < data_shape.size(); ++i)
    {
        if (!m_across_channels && i == 1)
        {
            element_count /= data_shape[i];
            continue;
        }
        reduction_axes.insert(i);
    }

    auto sum = make_shared<ngraph::op::Sum>(data, reduction_axes);
    auto element_count_node =
        op::Constant::create(data->get_element_type(), sum->get_shape(), vector<size_t>{element_count});
    auto mean = sum / element_count_node;
    mean = legacy_style_broadcast_for_binary_operation(data, mean, 1).at(1);

    auto mean_normalization = data - mean;

    return {mean_normalization};
}

shared_ptr<Node> op::MVN::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<MVN>(new_args.at(0), m_across_channels, m_normalize_variance, m_eps);
}
