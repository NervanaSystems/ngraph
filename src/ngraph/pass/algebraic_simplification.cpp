/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <memory>
#include <set>

#include "algebraic_simplification.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/pattern/matcher.hpp"

using namespace ngraph;

#define TI(x) std::type_index(typeid(x))

template <typename T>
static std::shared_ptr<pattern::Matcher>
    create_binary_matcher(std::shared_ptr<pattern::op::Label> label,
                          std::shared_ptr<pattern::op::Label> const_label)
{
    auto bcst_pred = [](std::shared_ptr<Node> n) {
        return std::dynamic_pointer_cast<op::Broadcast>(n) != nullptr;
    };

    auto bcst = std::make_shared<pattern::op::Skip>(const_label, bcst_pred);
    auto bcst_label = std::make_shared<pattern::op::Label>(bcst, nullptr, NodeVector{bcst});
    auto matcher =
        std::make_shared<pattern::Matcher>(std::make_shared<T>(label, bcst_label), nullptr);
    return matcher;
}

static std::shared_ptr<pattern::op::Label>
    get_broadcast_label(std::shared_ptr<pattern::Matcher> matcher)
{
    return std::dynamic_pointer_cast<pattern::op::Label>(matcher->pattern_node()->get_argument(1));
}

//`simplify_multiply` optimizes the following 4 *base* cases
//(8 cases in total including variants due to commutativity)
//
//a * 0 -> 0
//a * broadcast(0) -> broadcast(0)
//a * 1 -> a
//a * broadcast(1) -> a
static bool simplify_multiply(std::shared_ptr<Node> n)
{
    NGRAPH_DEBUG << "In simplify_multiply for " << n->get_name();
    auto iconst = ngraph::make_zero(element::i32, Shape{});
    auto label = std::make_shared<pattern::op::Label>(iconst);
    auto const_label_zero =
        std::make_shared<pattern::op::Label>(iconst, ngraph::is_zero, NodeVector{iconst});
    auto const_label_one =
        std::make_shared<pattern::op::Label>(iconst, ngraph::is_one, NodeVector{iconst});

    auto matcher_const_zero = create_binary_matcher<op::Multiply>(label, const_label_zero);
    auto matcher_const_one = create_binary_matcher<op::Multiply>(label, const_label_one);

    if (matcher_const_zero->match(n))
    {
        auto bcst_label = get_broadcast_label(matcher_const_zero);
        auto bcst_or_cnst = matcher_const_zero->get_pattern_map()[bcst_label];
        NGRAPH_DEBUG << " Replacing " << n->get_name() << " with " << bcst_or_cnst->get_name();
        ngraph::replace_node(n, bcst_or_cnst);
        return true;
    }

    if (matcher_const_one->match(n))
    {
        auto x = matcher_const_one->get_pattern_map()[label];
        NGRAPH_DEBUG << " Replacing " << n->get_name() << " with " << x->get_name();
        ngraph::replace_node(n, x);
        return true;
    }

    return false;
}

//`simplify_multiply` optimizes the following 2 *base* cases
//(4 cases in total including variants due to commutativity)
//
//a + 0 -> a
//a + broadcast(0) -> a
static bool simplify_add(std::shared_ptr<Node> n)
{
    NGRAPH_DEBUG << "In simplify_add for " << n->get_name();
    auto iconst = ngraph::make_zero(element::i32, Shape{});
    auto label = std::make_shared<pattern::op::Label>(iconst);
    auto const_label = std::make_shared<pattern::op::Label>(iconst, nullptr, NodeVector{iconst});
    auto matcher = create_binary_matcher<op::Add>(label, const_label);

    if (matcher->match(n))
    {
        auto pattern_map = matcher->get_pattern_map();
        auto x = pattern_map[label];
        auto cnst = pattern_map[const_label];
        NGRAPH_DEBUG << "Node " << n->get_name() << " matched \" arg + 0 \" \n"
                     << " arg : " << x->get_name() << " , const : " << cnst->get_name();

        if (ngraph::is_zero(cnst))
        {
            NGRAPH_DEBUG << " Replacing " << n->get_name() << " with " << x->get_name();
            ngraph::replace_node(n, x);
            return true;
        }
        else
        {
            NGRAPH_DEBUG << cnst->get_name() << " not equal to 0 ";
        }
    }
    return false;
}

static size_t reduction_shape_size(const AxisSet& axes, const Shape& shape)
{
    size_t prod = 1;
    for (auto axis : axes)
    {
        prod *= shape.at(axis);
    }

    return prod;
}

template <typename T>
static std::shared_ptr<Node>
    multiply_by(element::Type type, size_t multiplier, std::shared_ptr<op::Constant> cnst)
{
    T sum_cnst = static_cast<T>(cnst->get_vector<T>().at(0) * multiplier);
    return op::Constant::create<T>(type, Shape{}, {sum_cnst});
}

static std::shared_ptr<Node> get_sum_constant(std::shared_ptr<op::Constant> cnst, size_t multiplier)
{
    if (cnst->get_element_type() == element::i32)
    {
        return multiply_by<int>(cnst->get_element_type(), multiplier, cnst);
    }
    else if (cnst->get_element_type() == element::i8)
    {
        return multiply_by<char>(cnst->get_element_type(), multiplier, cnst);
    }
    else if (cnst->get_element_type() == element::f32)
    {
        return multiply_by<float>(cnst->get_element_type(), multiplier, cnst);
    }
    else if (cnst->get_element_type() == element::f64)
    {
        return multiply_by<double>(cnst->get_element_type(), multiplier, cnst);
    }

    return nullptr;
}

//`simplify_sum` optimizes the following case:
//sum(broadcast(scalar_constant), reduction_axes = ...) -> constant2 (or scalar constant)
//where constant2's values are equal to scalar_constant * shape_size(reduction_axes)
static bool simplify_sum(std::shared_ptr<Node> n)
{
    NGRAPH_DEBUG << "In simplify_sum for " << n->get_name();
    auto sum = std::dynamic_pointer_cast<op::Sum>(n);

    auto broadcast = std::dynamic_pointer_cast<op::Broadcast>(n->get_argument(0));
    if (!broadcast)
    {
        NGRAPH_DEBUG << n->get_name() << " isn't Broadcast";
        return false;
    }

    auto cnst = std::dynamic_pointer_cast<op::Constant>(broadcast->get_argument(0));
    if (!cnst || cnst->get_shape().size() > 0 /*not a scalar*/)
    {
        NGRAPH_DEBUG << broadcast->get_argument(0)->get_name() << " isn't a scalar constant";
        return false;
    }

    auto multiplier = reduction_shape_size(sum->get_reduction_axes(), broadcast->get_shape());
    auto sum_cnst = get_sum_constant(cnst, multiplier);

    //Unsupported type
    if (!sum_cnst)
    {
        NGRAPH_DEBUG << "unsupported type";
        return false;
    }

    if (sum->get_shape().size() > 0)
    {
        ngraph::AxisSet axes{};
        for (size_t i = 0; i < sum->get_shape().size(); i++)
        {
            axes.insert(i);
        }
        sum_cnst = std::make_shared<op::Broadcast>(sum_cnst, sum->get_shape(), axes);
    }

    ngraph::replace_node(n, sum_cnst);
    return true;
}

static std::unordered_map<std::type_index, std::function<bool(std::shared_ptr<Node>)>>
    initialize_const_values_to_ops()
{
    return std::unordered_map<std::type_index, std::function<bool(std::shared_ptr<Node>)>>({
        {TI(op::Add), simplify_add},
        {TI(op::Multiply), simplify_multiply},
        {TI(op::Sum), simplify_sum},
    });
}

static std::unordered_map<std::type_index, std::function<bool(std::shared_ptr<Node>)>>
    ops_to_const_values = initialize_const_values_to_ops();

bool ngraph::pass::AlgebraicSimplification::run_on_function(std::shared_ptr<ngraph::Function> f)
{
    bool replaced = false;
    for (auto n : f->get_ordered_ops())
    {
        if (n->is_output() || n->is_parameter())
        {
            continue;
        }

        const Node& node = *n;
        auto eh = ops_to_const_values.find(TI(node));
        if (eh == ops_to_const_values.end())
        {
            continue;
        }

        replaced = eh->second(n) || replaced;
    }
    return replaced;
}
