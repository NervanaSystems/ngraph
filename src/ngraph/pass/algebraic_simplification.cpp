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

#include <memory>
#include <set>

#include "algebraic_simplification.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

bool is_uniform_constant(const Input<Node>& input)
{
    bool rc = false;
    auto node = input.get_source_output().get_node();
    if (node->get_type_info() == op::Constant::type_info)
    {
        auto constant = as_type<op::Constant>(node);
        rc = constant->get_all_data_elements_bitwise_identical();
    }
    return rc;
}

extern template Shape ngraph::apply_permutation<Shape>(Shape input, AxisVector order);
template <typename T>
static shared_ptr<pattern::Matcher>
    create_binary_matcher(shared_ptr<pattern::op::Label> label,
                          shared_ptr<pattern::op::Label> const_label)
{
    auto bcst = make_shared<pattern::op::Skip>(const_label, pattern::has_class<op::Broadcast>());
    auto bcst_label = make_shared<pattern::op::Label>(bcst, nullptr, NodeVector{bcst});
    auto matcher = make_shared<pattern::Matcher>(make_shared<T>(label, bcst_label));
    return matcher;
}

//`simplify_concat` identifies slices-concat sequences
// that cancel each other. Namely it replaces subgraphs
// similar to the one below with `arg`
//
//                 +----------+
//            +----+slice(n/2..n)---+
// +-------+  |    +----------+    |  +-----------+
// |  arg  +--+                    +--+  concat   |
// +-------+  |    +----------+    |  +-----------+
//            +----+slice(0..n/2)---+
//                 +----------+
static bool simplify_concat(shared_ptr<Node> n)
{
    NGRAPH_DEBUG << "In simplify_concat for " << n->get_name();

    shared_ptr<Node> branch_tip;

    auto ltip = make_shared<pattern::op::Label>(element::i32, Shape{2, 1});

    auto pslice = make_shared<op::Slice>(ltip, Coordinate{0, 0}, Coordinate{2, 1}, Strides{1, 1});

    auto lslice = make_shared<pattern::op::Label>(pslice, nullptr, NodeVector{pslice});

    auto skip_reshape = make_shared<pattern::op::Skip>(lslice, pattern::has_class<op::Reshape>());

    auto matcher = make_shared<pattern::Matcher>(skip_reshape);

    Coordinate prev_lower_bounds;
    Shape prev_slice_shape;

    for (auto carg : n->get_arguments())
    {
        if (!matcher->match(carg))
        {
            NGRAPH_DEBUG << carg->get_name() << " doesn't match";
            return false;
        }

        auto slice = static_pointer_cast<op::Slice>(matcher->get_pattern_map()[lslice]);
        if (branch_tip)
        {
            if (branch_tip != matcher->get_pattern_map()[ltip])
            {
                NGRAPH_DEBUG << branch_tip->get_name() << " doesn't match "
                             << matcher->get_pattern_map()[ltip]->get_name();
                return false;
            }

            // slice chunks should be slice in the same order as slice nodes in concat's argument
            // list
            auto cur_lower_bounds = slice->get_lower_bounds();
            if (cur_lower_bounds < prev_lower_bounds)
            {
                NGRAPH_DEBUG << slice->get_name() << " is in the wrong order";
                return false;
            }
            prev_lower_bounds.assign(cur_lower_bounds.begin(), cur_lower_bounds.end());

            // slice shapes need to match
            if (slice->get_shape() != prev_slice_shape)
            {
                NGRAPH_DEBUG << slice->get_name()
                             << " doesn't match the shape of the previous slice";
                return false;
            }
        }
        else
        {
            branch_tip = matcher->get_pattern_map()[ltip];
            prev_lower_bounds.assign(slice->get_lower_bounds().begin(),
                                     slice->get_lower_bounds().end());
            prev_slice_shape.assign(slice->get_shape().begin(), slice->get_shape().end());
            NGRAPH_DEBUG << "setting branch_tip to " << branch_tip->get_name();
        }

        if (slice->get_users(true).size() > 1)
        {
            NGRAPH_DEBUG << slice->get_name() << " has more than one user";
            return false;
        }

        if (shape_size(slice->get_strides()) != 1)
        {
            NGRAPH_DEBUG << slice->get_name() << " is strided";
            return false;
        }

        // check that no other node uses slices and reshapes
        if (auto rcarg = as_type_ptr<op::Reshape>(carg))
        {
            auto default_shape = get_default_order(rcarg->get_argument(0)->get_shape());
            if (default_shape != rcarg->get_input_order())
            {
                NGRAPH_DEBUG << carg->get_name() << " reshape also does transposes";
                return false;
            }

            if (rcarg->get_users(true).size() > 1)
            {
                NGRAPH_DEBUG << rcarg->get_name() << " has more than one user";
                return false;
            }
        }
    }

    auto concat = static_pointer_cast<op::Concat>(n);
    auto concat_axis = concat->get_concatenation_axis();

    auto slice_shape = branch_tip->get_users(true).at(0)->get_shape();
    size_t slice_axis = numeric_limits<size_t>::max();

    auto btip_shape = branch_tip->get_shape();

    // slices should cover all elements
    if (shape_size(btip_shape) != shape_size(n->get_shape()))
    {
        NGRAPH_DEBUG << "The number of elements in Concat (" << shape_size(n->get_shape())
                     << ")  and the total of elements in slices (" << shape_size(btip_shape)
                     << ") don't match";
        return false;
    }

    for (size_t i = 0; i < btip_shape.size(); i++)
    {
        if (btip_shape[i] != slice_shape[i])
        {
            if (slice_axis != numeric_limits<size_t>::max())
            {
                // multi-axis slice + concat do not cancel
                return false;
            }
            slice_axis = i;
        }
    }

    if (slice_axis == numeric_limits<size_t>::max())
    {
        return false;
    }
    auto replacement = branch_tip;
    if (btip_shape != n->get_shape())
    {
        auto default_order = get_default_order(btip_shape);
        if (concat_axis == slice_axis)
        {
            // logical reshape only
            replacement = make_shared<op::Reshape>(branch_tip, default_order, concat->get_shape());
        }
        else
        {
            // axis reordering required
            auto transposed_shape = n->get_shape();

            if (btip_shape.size() >= transposed_shape.size())
            {
                AxisVector order = get_default_order(btip_shape);
                auto ax = order[slice_axis];
                order[slice_axis] = order[concat_axis];
                order[concat_axis] = ax;
                replacement = make_shared<op::Reshape>(branch_tip, order, transposed_shape);
            }
            else if (btip_shape.size() < transposed_shape.size())
            {
                // intermediate logical reshape
                AxisVector order = get_default_order(transposed_shape);
                auto ax = order[slice_axis];
                order[slice_axis] = order[concat_axis];
                order[concat_axis] = ax;
                auto output_shape = apply_permutation(transposed_shape, order);
                auto logical_reshape =
                    make_shared<op::Reshape>(branch_tip, default_order, output_shape);
                // transpose to final concatenated shape
                replacement = make_shared<op::Reshape>(logical_reshape, order, transposed_shape);
            }
        }
    }

    replace_node(n, replacement);
    return true;
}

static bool is_uniform_constant(const op::Constant* constant, int value)
{
    bool rc = false;
    if (constant && constant->get_all_data_elements_bitwise_identical())
    {
        switch (constant->get_element_type())
        {
        case ngraph::element::Type_t::undefined:
        {
            throw runtime_error("is_value type not supported");
        }
        case ngraph::element::Type_t::dynamic: { throw runtime_error("is_value type not supported");
        }
        case ngraph::element::Type_t::boolean: break;
        case ngraph::element::Type_t::bf16:
            rc = *static_cast<const bfloat16*>(constant->get_data_ptr()) ==
                 bfloat16(static_cast<float>(value));
            break;
        case ngraph::element::Type_t::f16:
            rc = *static_cast<const float16*>(constant->get_data_ptr()) ==
                 float16(static_cast<float>(value));
            break;
        case ngraph::element::Type_t::f32:
            rc = *static_cast<const float*>(constant->get_data_ptr()) == static_cast<float>(value);
            break;
        case ngraph::element::Type_t::f64:
            rc =
                *static_cast<const double*>(constant->get_data_ptr()) == static_cast<double>(value);
            break;
        case ngraph::element::Type_t::i8:
            rc =
                *static_cast<const int8_t*>(constant->get_data_ptr()) == static_cast<int8_t>(value);
            break;
        case ngraph::element::Type_t::i16:
            rc = *static_cast<const int16_t*>(constant->get_data_ptr()) ==
                 static_cast<int16_t>(value);
            break;
        case ngraph::element::Type_t::i32:
            rc = *static_cast<const int32_t*>(constant->get_data_ptr()) ==
                 static_cast<int32_t>(value);
            break;
        case ngraph::element::Type_t::i64:
            rc = *static_cast<const int64_t*>(constant->get_data_ptr()) ==
                 static_cast<int64_t>(value);
            break;
        case ngraph::element::Type_t::u1: throw runtime_error("is_value type not supported");
        case ngraph::element::Type_t::u8:
            rc = *static_cast<const uint8_t*>(constant->get_data_ptr()) ==
                 static_cast<uint8_t>(value);
            break;
        case ngraph::element::Type_t::u16:
            rc = *static_cast<const uint16_t*>(constant->get_data_ptr()) ==
                 static_cast<uint16_t>(value);
            break;
        case ngraph::element::Type_t::u32:
            rc = *static_cast<const uint32_t*>(constant->get_data_ptr()) ==
                 static_cast<uint32_t>(value);
            break;
        case ngraph::element::Type_t::u64:
            rc = *static_cast<const uint64_t*>(constant->get_data_ptr()) ==
                 static_cast<uint64_t>(value);
            break;
        }
    }
    return rc;
}

static shared_ptr<op::Constant> get_constant(shared_ptr<Node> op)
{
    set<Node::type_info_t> nomath = {op::Broadcast::type_info, op::Reshape::type_info};
    while (nomath.find(op->get_type_info()) != nomath.end())
    {
        op = op->get_input_node_shared_ptr(0);
    }
    return as_type_ptr<op::Constant>(op);
}

static bool is_input_uniform_constant(shared_ptr<Node> op,
                                      int constant_value,
                                      shared_ptr<Node>& constant,
                                      shared_ptr<Node>& value)
{
    bool rc = false;
    auto c = get_constant(op->get_input_node_shared_ptr(0));
    if (is_uniform_constant(c.get(), constant_value))
    {
        constant = op->get_input_node_shared_ptr(0);
        value = op->get_input_node_shared_ptr(1);
        rc = true;
    }
    else
    {
        c = get_constant(op->get_input_node_shared_ptr(1));
        if (is_uniform_constant(c.get(), constant_value))
        {
            constant = op->get_input_node_shared_ptr(1);
            value = op->get_input_node_shared_ptr(0);
            rc = true;
        }
    }
    return rc;
}

//`simplify_multiply` optimizes the following 4 *base* cases
//(8 cases in total including variants due to commutativity)
//
// a * 0 -> 0
// a * broadcast(0) -> broadcast(0)
// a * 1 -> a
// a * broadcast(1) -> a
static bool simplify_multiply(shared_ptr<Node> n)
{
    bool rc = false;
    auto multiply = as_type_ptr<op::Multiply>(n);
    if (multiply)
    {
        shared_ptr<Node> constant;
        shared_ptr<Node> value;
        if (is_input_uniform_constant(multiply, 0, constant, value))
        {
            replace_node(multiply, constant);
            rc = true;
        }
        else
        {
            if (is_input_uniform_constant(multiply, 1, constant, value))
            {
                replace_node(multiply, value);
                rc = true;
            }
        }
    }

    return rc;
}

//`simplify_add` optimizes the following 2 *base* cases
//(4 cases in total including variants due to commutativity)
//
// a + 0 -> a
// a + broadcast(0) -> a
static bool simplify_add(shared_ptr<Node> n)
{
    bool rc = false;
    auto add = as_type_ptr<op::Add>(n);
    if (add)
    {
        shared_ptr<Node> constant;
        shared_ptr<Node> value;
        if (is_input_uniform_constant(add, 0, constant, value))
        {
            replace_node(add, value);
            rc = true;
        }
    }

    return rc;
}

//`simplify_log` optimizes `log(exp(x)/y)` into `x - log(y)`
static bool simplify_log(shared_ptr<Node> n)
{
    if (auto div = as_type_ptr<op::Divide>(n->input_value(0).get_node_shared_ptr()))
    {
        if (auto exp = as_type_ptr<op::Exp>(div->input_value(0).get_node_shared_ptr()))
        {
            auto denom = div->get_argument(1);
            auto diff =
                make_shared<op::Subtract>(exp->get_argument(0), make_shared<op::Log>(denom));
            replace_node(n, diff);
            return true;
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
static shared_ptr<Node>
    multiply_by(element::Type type, size_t multiplier, shared_ptr<op::Constant> cnst)
{
    T sum_cnst = static_cast<T>(cnst->get_vector<T>().at(0) * multiplier);
    return op::Constant::create<T>(type, Shape{}, {sum_cnst});
}

template <typename T>
static shared_ptr<Node> pow_by(element::Type type, size_t multiplier, shared_ptr<op::Constant> cnst)
{
    T prod = static_cast<T>(1);
    T val = cnst->get_vector<T>().at(0);
    for (size_t i = 0; i < multiplier; i++)
    {
        prod *= val;
    }
    return op::Constant::create<T>(type, Shape{}, {prod});
}

static shared_ptr<Node> get_sum_constant(shared_ptr<op::Constant> cnst, size_t multiplier)
{
    if (cnst->get_element_type() == element::i32)
    {
        return multiply_by<int>(cnst->get_element_type(), multiplier, cnst);
    }
    else if (cnst->get_element_type() == element::i8)
    {
        return multiply_by<signed char>(cnst->get_element_type(), multiplier, cnst);
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

static shared_ptr<Node> get_prod_constant(shared_ptr<op::Constant> cnst, size_t multiplier)
{
    if (cnst->get_element_type() == element::i32)
    {
        return pow_by<int>(cnst->get_element_type(), multiplier, cnst);
    }
    else if (cnst->get_element_type() == element::i8)
    {
        return pow_by<signed char>(cnst->get_element_type(), multiplier, cnst);
    }
    else if (cnst->get_element_type() == element::f32)
    {
        return pow_by<float>(cnst->get_element_type(), multiplier, cnst);
    }
    else if (cnst->get_element_type() == element::f64)
    {
        return pow_by<double>(cnst->get_element_type(), multiplier, cnst);
    }

    return nullptr;
}

//`simplify_reduction` optimizes the following case:
// sum(broadcast(scalar_constant), reduction_axes = ...) -> constant2 (or scalar constant)
// where constant2's values are equal to scalar_constant * shape_size(reduction_axes)
// product(broadcast(scalar_constant), reduction_axes = ...) -> constant2 (or scalar constant)
// where constant2's values are equal to scalar_constant ^ shape_size(reduction_axes)
template <typename T, shared_ptr<Node> (*F)(shared_ptr<op::Constant> cnst, size_t multiplier)>
static bool simplify_reduction(shared_ptr<Node> n)
{
    NGRAPH_DEBUG << "In simplify_reduction for " << n->get_name();
    auto reduction = static_pointer_cast<T>(n);

    auto broadcast = as_type_ptr<op::Broadcast>(n->input_value(0).get_node_shared_ptr());
    if (!broadcast)
    {
        NGRAPH_DEBUG << n->get_name() << " isn't Broadcast";
        return false;
    }

    auto cnst = as_type_ptr<op::Constant>(broadcast->input_value(0).get_node_shared_ptr());
    if (!cnst || cnst->get_shape().size() > 0 /*not a scalar*/)
    {
        NGRAPH_DEBUG << broadcast->get_argument(0)->get_name() << " isn't a scalar constant";
        return false;
    }

    auto multiplier = reduction_shape_size(reduction->get_reduction_axes(), broadcast->get_shape());
    auto reduction_cnst = F(cnst, multiplier);

    // Unsupported type
    if (!reduction_cnst)
    {
        NGRAPH_DEBUG << "unsupported type";
        return false;
    }

    if (reduction->get_shape().size() > 0)
    {
        AxisSet axes{};
        for (size_t i = 0; i < reduction->get_shape().size(); i++)
        {
            axes.insert(i);
        }
        reduction_cnst = make_shared<op::Broadcast>(reduction_cnst, reduction->get_shape(), axes);
    }

    replace_node(n, reduction_cnst);
    return true;
}

static unordered_map<NodeTypeInfo, function<bool(shared_ptr<Node>)>> initialize_ops_to_simplifiers()
{
    return unordered_map<NodeTypeInfo, function<bool(shared_ptr<Node>)>>(
        {{op::Add::type_info, simplify_add},
         {op::Multiply::type_info, simplify_multiply},
         {op::Concat::type_info, simplify_concat},
         {op::Sum::type_info,
          function<bool(shared_ptr<Node>)>{simplify_reduction<op::Sum, get_sum_constant>}},
         {op::Product::type_info,
          function<bool(shared_ptr<Node>)>{simplify_reduction<op::Product, get_prod_constant>}},
         {op::Log::type_info, simplify_log}});
}

static unordered_map<NodeTypeInfo, function<bool(shared_ptr<Node>)>> ops_to_simplifiers =
    initialize_ops_to_simplifiers();

bool pass::AlgebraicSimplification::run_on_function(shared_ptr<Function> f)
{
    bool replaced = false;
    for (auto n : f->get_ordered_ops())
    {
        if (n->is_output() || n->is_parameter())
        {
            continue;
        }

        auto eh = ops_to_simplifiers.find(n->get_type_info());
        if (eh != ops_to_simplifiers.end())
        {
            replaced |= eh->second(n);
        }
    }
    return replaced;
}
