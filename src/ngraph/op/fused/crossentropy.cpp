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
#include "ngraph/op/fused/crossentropy.hpp"

#include "ngraph/builder/make_constant.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/util/broadcasting.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::CrossEntropy::type_info;

op::CrossEntropy::CrossEntropy(const Output<Node>& arg1,
                               const Output<Node>& arg2,
                               bool soft_label,
                               int64_t ignore_index)
    : FusedOp({arg1, arg2})
    , m_soft_label(soft_label)
    , m_ignore_index(ignore_index)
{
    constructor_validate_and_infer_types();
}

NodeVector op::CrossEntropy::decompose_op() const
{
    auto input_to_normalize = input_value(0);
    auto labels = input_value(1);
    auto reduction_axis = input_to_normalize.get_shape().size() - 1;

    auto create_mask = [&]() -> std::shared_ptr<ngraph::Node> {
        // ignore mask
        auto mask_constant = ngraph::op::Constant::create(
            labels.get_element_type(), labels.get_shape(), {m_ignore_index});
        auto not_equal = std::make_shared<ngraph::op::NotEqual>(labels, mask_constant);
        auto convert =
            std::make_shared<ngraph::op::Convert>(not_equal, input_to_normalize.get_element_type());
        auto reshape = std::make_shared<ngraph::op::Reshape>(
            convert, AxisVector{0, 1}, Shape{convert->get_shape().at(0), 1});
        return reshape;
    };

    auto create_xe = [&](const Output<Node>& one_hot, const Output<Node>& input) {
        auto node_log = std::make_shared<ngraph::op::Log>(input);
        auto node_mul = one_hot * node_log;
        auto node_sum = std::make_shared<ngraph::op::Sum>(
            node_mul, AxisSet{static_cast<size_t>(reduction_axis)});
        return -node_sum;
    };

    // mask
    std::shared_ptr<ngraph::Node> mask = create_mask();

    if (m_soft_label)
    {
        // insert dtype conversion if required
        if (labels.get_element_type() != input_to_normalize.get_element_type())
        {
            labels = std::make_shared<ngraph::op::Convert>(labels,
                                                           input_to_normalize.get_element_type());
        }
        auto reshape_labels = std::make_shared<ngraph::op::Reshape>(
            labels, AxisVector{0, 1}, Shape{labels.get_shape().at(0)});
        auto broadcast_labels = std::make_shared<ngraph::op::Broadcast>(
            reshape_labels,
            input_to_normalize.get_shape(),
            AxisSet{input_to_normalize.get_shape().size() - 1});
        auto xe = create_xe(broadcast_labels, input_to_normalize);
        auto reshape_xe = std::make_shared<ngraph::op::Reshape>(
            xe, AxisVector{0}, Shape{xe->get_shape().at(0), 1});
        return {reshape_xe};
    }
    else
    {
        // we will have one_hot encoding on labels if softmax_lables = false
        size_t one_hot_axis = input_to_normalize.get_shape().size() - 1;
        size_t softmax_axis = input_to_normalize.get_shape().size() - 1;
        auto reshape_labels =
            make_shared<op::Reshape>(labels, AxisVector{0, 1}, Shape{labels.get_shape().at(0)});
        auto one_hot_labels = std::make_shared<ngraph::op::OneHot>(
            reshape_labels, input_to_normalize.get_shape(), one_hot_axis);
        auto convert_one_hot = std::make_shared<ngraph::op::Convert>(
            one_hot_labels, input_to_normalize.get_element_type());

        // calculate loss
        auto xe = create_xe(convert_one_hot, input_to_normalize);
        auto reshape_xe = std::make_shared<ngraph::op::Reshape>(
            xe, AxisVector{0}, Shape{xe->get_shape().at(0), 1});
        if (m_ignore_index > 0)
        {
            return {reshape_xe * mask};
        }
        return {reshape_xe};
    }
}

shared_ptr<Node> op::CrossEntropy::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<CrossEntropy>(new_args.at(0), new_args.at(1), m_soft_label, m_ignore_index);
}

constexpr NodeTypeInfo op::CrossEntropyBackprop::type_info;

op::CrossEntropyBackprop::CrossEntropyBackprop(const Output<Node>& input,
                                               const Output<Node>& labels,
                                               const Output<Node>& delta,
                                               bool soft_label,
                                               int64_t ignore_index)
    : FusedOp({input, labels, delta})
    , m_soft_label(soft_label)
    , m_ignore_index(ignore_index)
{
    constructor_validate_and_infer_types();
}

void op::CrossEntropyBackprop::pre_validate_and_infer_types()
{
    element::Type input_element_type = get_input_element_type(0);

    NODE_VALIDATION_CHECK(this,
                          input_element_type.is_dynamic() || input_element_type.is_real(),
                          "Argument element type must be f16, bf16, f32, f64 or dynamic (got ",
                          input_element_type,
                          ").");
}

shared_ptr<Node> op::CrossEntropyBackprop::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<CrossEntropyBackprop>(
        new_args.at(0), new_args.at(1), new_args.at(2), m_soft_label, m_ignore_index);
}

NodeVector op::CrossEntropyBackprop::decompose_op() const
{
    auto input = input_value(0);
    auto labels = input_value(1);
    auto delta = input_value(2);
    auto rank = input.get_shape().size();

    size_t one_hot_axis = delta.get_shape().size() - 1;

    // always reduces the sum on the last axis
    auto reduction_axis = delta.get_shape().size() - 1;

    // mask
    std::shared_ptr<ngraph::Node> mask = nullptr;

    // remove trailing ones from delta
    auto delta_reshape = std::make_shared<ngraph::op::Reshape>(
        delta, AxisVector{0, 1}, Shape{delta.get_shape().at(0)});
    auto delta_bcast = std::make_shared<ngraph::op::Broadcast>(
        delta_reshape, input.get_shape(), AxisSet{rank - 1});

    if (!m_soft_label)
    {
        // ignore mask
        if (m_ignore_index > 0)
        {
            auto mask_constant =
                ngraph::op::Constant::create(element::i64, labels.get_shape(), {m_ignore_index});
            auto not_equal = std::make_shared<ngraph::op::NotEqual>(labels, mask_constant);
            auto convert =
                std::make_shared<ngraph::op::Convert>(not_equal, input.get_element_type());
            auto reshape = std::make_shared<ngraph::op::Reshape>(
                convert, AxisVector{0, 1}, Shape{convert->get_shape().at(0)});
            mask = std::make_shared<ngraph::op::Broadcast>(
                reshape, input.get_shape(), AxisSet{rank - 1});
        }
        // one hot encoding of labels
        auto reshape_labels =
            make_shared<op::Reshape>(labels, AxisVector{0, 1}, Shape{labels.get_shape().at(0)});
        auto one_hot =
            std::make_shared<ngraph::op::OneHot>(reshape_labels, input.get_shape(), one_hot_axis);
        labels = std::make_shared<ngraph::op::Convert>(one_hot, input.get_element_type());
    }

    std::shared_ptr<ngraph::Node> xe_grad =
        std::make_shared<ngraph::op::Divide>(-labels * delta_bcast, input);

    if (!m_soft_label && m_ignore_index > 0)
    {
        xe_grad = xe_grad * mask;
    }
    return {xe_grad};
}
