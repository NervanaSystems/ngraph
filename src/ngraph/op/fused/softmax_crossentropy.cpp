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
#include "ngraph/op/fused/softmax_crossentropy.hpp"

#include "ngraph/builder/make_constant.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/util/broadcasting.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::SoftmaxCrossEntropy::type_info;

op::SoftmaxCrossEntropy::SoftmaxCrossEntropy(const Output<Node>& arg1,
                                             const Output<Node>& arg2,
                                             const AxisSet& reduction_axes)
    : FusedOp({arg1, arg2})
    , m_summation_axis{reduction_axes}
{
    constructor_validate_and_infer_types();
}

NodeVector op::SoftmaxCrossEntropy::decompose_op() const
{
    auto input_to_normalize = input_value(0);
    auto one_hot_labels = input_value(1);

    auto max_xj = std::make_shared<ngraph::op::Max>(input_to_normalize, m_summation_axis);
    auto broadcast_max_xj =
        std::make_shared<ngraph::op::Broadcast>(max_xj, input_to_normalize.get_shape(), AxisSet{1});
    auto subtract = std::make_shared<ngraph::op::Subtract>(input_to_normalize, broadcast_max_xj);
    auto exp = std::make_shared<ngraph::op::Exp>(subtract);

    auto sum_over_j = std::make_shared<ngraph::op::Sum>(exp, m_summation_axis);
    auto log_sum_over_j = std::make_shared<ngraph::op::Log>(sum_over_j);

    auto subtract_max_xj_from_input =
        std::make_shared<ngraph::op::Subtract>(input_to_normalize, broadcast_max_xj);
    auto broadcast_log = std::make_shared<ngraph::op::Broadcast>(
        log_sum_over_j, subtract_max_xj_from_input->get_shape(), AxisSet{1});
    auto subtract_max_xj_from_input_from_log_sum_over_j =
        std::make_shared<ngraph::op::Subtract>(subtract_max_xj_from_input, broadcast_log);
    auto multiply = std::make_shared<ngraph::op::Multiply>(
        one_hot_labels, subtract_max_xj_from_input_from_log_sum_over_j);
    auto sum_over_k = std::make_shared<ngraph::op::Sum>(multiply, m_summation_axis);
    auto negate_summation = std::make_shared<ngraph::op::Negative>(sum_over_k);

    return {negate_summation};
}

shared_ptr<Node> op::SoftmaxCrossEntropy::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<SoftmaxCrossEntropy>(new_args.at(0), new_args.at(1), m_summation_axis);
}
