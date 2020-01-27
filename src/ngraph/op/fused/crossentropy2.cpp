//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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
#include "ngraph/op/fused/crossentropy2.hpp"

#include "ngraph/builder/make_constant.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/equal.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/util/broadcasting.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::CrossEntropy2::type_info;

op::CrossEntropy2::CrossEntropy2(const Output<Node>& arg1,
                                 const Output<Node>& arg2,
                                 bool soft_label,
                                 int64_t ignore_index)
    : FusedOp({arg1, arg2})
    , m_soft_label(soft_label)
    , m_ignore_index(ignore_index)
{
    constructor_validate_and_infer_types();
}
// create mask based on ignore_index
static std::shared_ptr<ngraph::Node>
    create_mask(Output<Node> labels, Output<Node> input, int64_t ignore_index)
{
    auto mask_constant =
        ngraph::op::Constant::create(labels.get_element_type(), labels.get_shape(), {ignore_index});
    auto not_equal = std::make_shared<ngraph::op::NotEqual>(labels, mask_constant);
    auto convert = std::make_shared<ngraph::op::Convert>(not_equal, input.get_element_type());
    return convert;
}

NodeVector op::CrossEntropy2::decompose_op() const
{
    auto input = input_value(0);
    auto labels = input_value(1);
    auto reduction_axis = input.get_shape().size() - 1;

    auto reshape = [&](const Output<Node>& input, ngraph::Shape shape) {
        const auto reshape_pattern = op::Constant::create(element::i64, Shape{shape.size()}, shape);
        std::shared_ptr<ngraph::Node> input_reshape =
            std::make_shared<op::v1::Reshape>(input, reshape_pattern, false);
        return input_reshape;
    };

    auto create_xe = [&](const Output<Node>& one_hot, const Output<Node>& input) {
        auto node_log = std::make_shared<op::Log>(input);
        auto node_mul = one_hot * node_log;
        auto node_sum =
            std::make_shared<op::Sum>(node_mul, AxisSet{static_cast<size_t>(reduction_axis)});
        auto input_shape = input.get_shape();
        input_shape.back() = 1;
        auto node_sum_reshape = reshape(-node_sum, input_shape);
        return node_sum_reshape;
    };

    auto create_one_hot = [&](const Output<Node>& label, const Output<Node>& x) {
        auto label_shape = label.get_shape();
        auto x_shape = x.get_shape();
        auto x_shape_size = x.get_shape().size() - 1;
        if (label_shape.back() == 1 && label_shape.size() > 1)
        {
            label_shape.pop_back();
            std::shared_ptr<ngraph::Node> X = reshape(label, label_shape);
            return std::make_shared<ngraph::op::OneHot>(X, x_shape, x_shape_size);
        }
        return std::make_shared<ngraph::op::OneHot>(label, x_shape, x_shape_size);
    };

    auto one_hot_shape = input.get_shape();
    auto rank = one_hot_shape.size() - 1;
    auto label_shape = labels.get_shape();

    std::shared_ptr<ngraph::Node> one_hot_labels = create_one_hot(labels, input);
    auto input_type = input.get_element_type();
    one_hot_labels = std::make_shared<op::Convert>(one_hot_labels, input_type);

    auto xe = create_xe(one_hot_labels, input);
    auto mask = create_mask(labels, xe, m_ignore_index);
    mask = std::make_shared<op::Convert>(mask, input_type);

    auto mask_shape = mask->get_shape();
    auto xe_reshape = reshape(xe, mask_shape);

    xe_reshape = xe_reshape * mask;

    auto node_sum = std::make_shared<op::Sum>(one_hot_labels * input, ngraph::AxisSet{rank});
    auto sum_reshape = reshape(node_sum, mask_shape);

    auto matchx = mask * sum_reshape;

    auto xe_shape = xe->get_shape();
    auto match_reshape = reshape(matchx, xe_shape);

    return {match_reshape, input.get_node_shared_ptr(), xe_reshape};
}

shared_ptr<Node> op::CrossEntropy2::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<CrossEntropy2>(new_args.at(0), new_args.at(1), m_soft_label, m_ignore_index);
}

void op::CrossEntropy2::pre_validate_and_infer_types()
{
    bool is_input_dynamic = false;

    for (size_t i = 0; i < get_input_size(); ++i)
    {
        if (get_input_partial_shape(i).is_dynamic())
        {
            is_input_dynamic = true;
            break;
        }
    }

    if (is_input_dynamic)
    {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
        set_output_type(1, get_input_element_type(0), PartialShape::dynamic());
        set_output_type(2, get_input_element_type(0), PartialShape::dynamic());
    }
}

constexpr NodeTypeInfo op::CrossEntropy2Backprop::type_info;

op::CrossEntropy2Backprop::CrossEntropy2Backprop(const Output<Node>& input,
                                                 const Output<Node>& labels,
                                                 const Output<Node>& x,
                                                 const Output<Node>& dy,
                                                 bool soft_label,
                                                 int64_t ignore_index)
    : FusedOp({input, labels, x, dy})
    , m_soft_label(soft_label)
    , m_ignore_index(ignore_index)
{
    constructor_validate_and_infer_types();
}

void op::CrossEntropy2Backprop::pre_validate_and_infer_types()
{
    bool is_input_dynamic = false;

    for (size_t i = 0; i < get_input_size(); ++i)
    {
        if (get_input_partial_shape(i).is_dynamic())
        {
            is_input_dynamic = true;
            break;
        }
    }

    if (is_input_dynamic)
    {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
    }
}

shared_ptr<Node> op::CrossEntropy2Backprop::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<CrossEntropy2Backprop>(new_args.at(0),
                                              new_args.at(1),
                                              new_args.at(2),
                                              new_args.at(3),
                                              m_soft_label,
                                              m_ignore_index);
}

NodeVector op::CrossEntropy2Backprop::decompose_op() const
{
    auto matchx = input_value(0);
    auto label = input_value(1);
    auto x = input_value(2);
    auto dy = input_value(3);

    auto reshape = [&](const Output<Node>& input, ngraph::Shape shape) {
        std::vector<size_t> input_order(input.get_shape().size());
        std::iota(std::begin(input_order), std::end(input_order), 0);
        std::shared_ptr<ngraph::Node> reshape =
            std::make_shared<op::Reshape>(input, ngraph::AxisVector(input_order), shape);
        return reshape;
    };

    auto create_one_hot = [&](const Output<Node>& label, const Output<Node>& x) {
        auto label_shape = label.get_shape();
        auto x_shape = x.get_shape();
        auto x_shape_size = x.get_shape().size() - 1;
        if (label_shape.back() == 1 && label_shape.size() > 1)
        {
            label_shape.pop_back();
            const auto reshape_pattern =
                op::Constant::create(element::i64, Shape{label_shape.size()}, label_shape);
            std::shared_ptr<ngraph::Node> X =
                std::make_shared<op::v1::Reshape>(label, reshape_pattern, false);
            return std::make_shared<ngraph::op::OneHot>(X, x_shape, x_shape_size);
        }
        return std::make_shared<ngraph::op::OneHot>(label, x_shape, x_shape_size);
    };

    auto matchx_shape = matchx.get_shape();
    auto label_shape = label.get_shape();
    auto x_shape = x.get_shape();
    auto dy_shape = dy.get_shape();
    if (matchx_shape.back() == 1 && matchx_shape.size() > 1)
    {
        matchx_shape.pop_back();
        matchx = reshape(matchx, matchx_shape);
    }

    if (label_shape.back() == 1 && label_shape.size() > 1)
    {
        label_shape.pop_back();
        label = reshape(label, label_shape);
    }

    if (x_shape.back() == 1 && x_shape.size() > 1)
    {
        x_shape.pop_back();
        x = reshape(x, x_shape);
    }

    if (dy_shape.back() == 1 && dy_shape.size() > 1)
    {
        dy_shape.pop_back();
        dy = reshape(dy, dy_shape);
    }

    auto rank = x_shape.size();
    auto x_type = x.get_element_type();

    std::shared_ptr<ngraph::Node> one_hot_labels = create_one_hot(label, x);
    one_hot_labels = std::make_shared<op::Convert>(one_hot_labels, x_type);

    auto mask = create_mask(label, x, m_ignore_index);
    mask = std::make_shared<op::Convert>(mask, x_type);

    auto zero = op::Constant::create(matchx.get_element_type(), matchx.get_shape(), {0});
    auto one = op::Constant::create(matchx.get_element_type(), matchx.get_shape(), {1});

    auto is_zero = std::make_shared<op::Equal>(matchx, zero);
    matchx = std::make_shared<ngraph::op::Select>(is_zero, one, matchx);

    auto dy_bcast =
        std::make_shared<ngraph::op::Broadcast>(mask * dy, x_shape, ngraph::AxisSet{rank - 1});

    auto matchx_bcast =
        std::make_shared<ngraph::op::Broadcast>(matchx, x_shape, ngraph::AxisSet{rank - 1});

    auto xe_grad = -dy_bcast * one_hot_labels / matchx_bcast;
    return {xe_grad};
}
