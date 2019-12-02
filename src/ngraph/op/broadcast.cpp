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

#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/util/broadcasting.hpp"
#include "ngraph/partial_shape.hpp"

#include <numeric>

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v1::Broadcast::type_info;

op::v1::Broadcast::Broadcast(const Output<Node>& arg,
                             const Output<Node>& target_shape,
                             const Output<Node>& axes_mapping,
                             const AutoBroadcastSpec& broadcast_spec)
    : Op({arg, target_shape, axes_mapping})
    , m_broadcast_spec(broadcast_spec)
{
    constructor_validate_and_infer_types();
}

op::v1::Broadcast::Broadcast(const Output<Node>& arg,
                             const Output<Node>& target_shape,
                             const AutoBroadcastSpec& broadcast_spec)
    : Op({arg, target_shape, op::Constant::create(element::u8, Shape{}, {0})->output(0)})
    , m_broadcast_spec(broadcast_spec)
{
    constructor_validate_and_infer_types();
}

bool op::v1::Broadcast::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("broadcast_spec", m_broadcast_spec);
    return true;
}

std::pair<bool, AxisSet> op::v1::Broadcast::get_broadcast_axes() const
{
    AxisSet broadcast_axes;
    bool axes_known = false;

    if (m_broadcast_spec.m_type == AutoBroadcastType::NONE)
    {
        if (input(1).get_partial_shape().is_static() &&
            input_value(2).get_node_shared_ptr()->is_constant())
        {
            auto target_shape = input(1).get_shape();
            NGRAPH_CHECK(target_shape.size() == 1);
            auto axes_mapping_val =
                static_pointer_cast<op::Constant>(input_value(2).get_node_shared_ptr())
                    ->get_axis_vector_val();

            std::vector<size_t> axes(target_shape[0]);
            std::iota(axes.begin(), axes.end(), 0);
            for (auto i = axes_mapping_val.rbegin(); i != axes_mapping_val.rend(); ++i)
            {
                axes.erase(axes.begin() + *i);
            }
            broadcast_axes.insert(axes.begin(), axes.end());
            axes_known = true;
        }
    }
    else if (m_broadcast_spec.m_type == AutoBroadcastType::NUMPY ||
             m_broadcast_spec.m_type == AutoBroadcastType::PDPD)
    {
        if (input(0).get_partial_shape().is_static() &&
            input_value(1).get_node_shared_ptr()->is_constant())
        {
            auto arg_shape = input(0).get_shape();
            auto target_shape =
                static_pointer_cast<op::Constant>(input_value(1).get_node_shared_ptr())
                    ->get_shape_val();
            auto start_axis = (m_broadcast_spec.m_type == AutoBroadcastType::PDPD)
                                  ? m_broadcast_spec.m_axis
                                  : target_shape.size() - arg_shape.size();
            NGRAPH_CHECK(start_axis >= 0);
            for (size_t i = 0; i < target_shape.size(); i++)
            {
                if (i < start_axis || target_shape[i] != arg_shape[i - start_axis])
                {
                    broadcast_axes.insert(i);
                }
            }
            axes_known = true;
        }
    }
    else
    {
        throw ngraph_error("Unknown autobroadcast type");
    }

    return std::make_pair(axes_known, broadcast_axes);
}

void op::v1::Broadcast::validate_and_infer_types()
{
    // shape node should have integer data type. For now we only allow i64
    auto shape_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          shape_et.compatible(element::Type_t::i64),
                          "Broadcast shape must have element type i64, but has ",
                          shape_et);

    // shape node should produce a one dimensional shape.
    auto broadcast_shape_rank = get_input_partial_shape(1).rank();
    NODE_VALIDATION_CHECK(this,
                          broadcast_shape_rank.compatible(1),
                          "Broadcast shape rank must be 1, but has ",
                          broadcast_shape_rank);

    if (m_broadcast_spec.m_type == AutoBroadcastType::NONE)
    {
        // axes_mapping node should have integer data type. For now we only allow i64
        auto axes_et = get_input_element_type(2);
        NODE_VALIDATION_CHECK(this,
                              axes_et.compatible(element::Type_t::i64),
                              "Broadcast axes must have element type i64, but has ",
                              axes_et);

        // axes_mapping node should produce a one dimensional shape.
        auto axes_shape_rank = get_input_partial_shape(2).rank();
        NODE_VALIDATION_CHECK(this,
                              axes_shape_rank.compatible(1),
                              "Broadcast axes rank must be 1, but has ",
                              axes_shape_rank);
    }

    PartialShape result_shape{PartialShape::dynamic()};
    if (input_value(1).get_node_shared_ptr()->is_constant())
    {
        result_shape = static_pointer_cast<op::Constant>(input_value(1).get_node_shared_ptr())
                           ->get_shape_val();
    }

    if (m_broadcast_spec.m_type == AutoBroadcastType::NONE)
    {
        // Validate axes_mapping
        if (input(0).get_partial_shape().is_static() && input(1).get_partial_shape().is_static() &&
            input(2).get_partial_shape().is_static())
        {
            auto arg_shape = input(0).get_shape();
            auto axes_shape = input(2).get_shape();

            // Rank(arg_shape) == shape_size(axes_mapping)
            NODE_VALIDATION_CHECK(this,
                                  shape_size(axes_shape) == arg_shape.size(),
                                  "Broadcast axes_mapping shape ",
                                  axes_shape,
                                  " doesn't match rank of input tensor ",
                                  arg_shape.size());

            if (input_value(1).get_node_shared_ptr()->is_constant() &&
                input_value(2).get_node_shared_ptr()->is_constant())
            {
                auto target_shape =
                    static_pointer_cast<op::Constant>(input_value(1).get_node_shared_ptr())
                        ->get_shape_val();
                auto axes_mapping_val =
                    static_pointer_cast<op::Constant>(input_value(2).get_node_shared_ptr())
                        ->get_axis_vector_val();
                // axes_mapping needs to be in sorted order
                NODE_VALIDATION_CHECK(
                    this,
                    std::is_sorted(axes_mapping_val.begin(), axes_mapping_val.end()),
                    "Broadcast doesn't permit transposes. axes_mapping ",
                    axes_mapping_val,
                    " not in sorted order");

                for (size_t i = 0; i < axes_mapping_val.size(); i++)
                {
                    NODE_VALIDATION_CHECK(this,
                                          axes_mapping_val[i] < target_shape.size(),
                                          "Broadcast axes_mapping[",
                                          i,
                                          "]: ",
                                          axes_mapping_val[i],
                                          " exceeds target rank ",
                                          target_shape.size());

                    NODE_VALIDATION_CHECK(this,
                                          target_shape[axes_mapping_val[i]] == arg_shape[i],
                                          "Broadcast target[axes_mapping[",
                                          i,
                                          "]]",
                                          " Expected ",
                                          arg_shape[i],
                                          ". Got ",
                                          target_shape[axes_mapping_val[i]]);
                }
            }
        }
    }
    else if (m_broadcast_spec.m_type == AutoBroadcastType::NUMPY ||
             m_broadcast_spec.m_type == AutoBroadcastType::PDPD)
    {
        if (input(0).get_partial_shape().is_static() && input(1).get_partial_shape().is_static())
        {
            auto arg_shape = input(0).get_shape();

            if (input_value(1).get_node_shared_ptr()->is_constant())
            {
                auto target_shape =
                    static_pointer_cast<op::Constant>(input_value(1).get_node_shared_ptr())
                        ->get_shape_val();
                auto start_axis = (m_broadcast_spec.m_type == AutoBroadcastType::PDPD)
                                      ? m_broadcast_spec.m_axis
                                      : target_shape.size() - arg_shape.size();
                NODE_VALIDATION_CHECK(this,
                                      start_axis >= 0,
                                      "Broadcast target_shape has smaller rank ",
                                      target_shape.size(),
                                      " than arg shape ",
                                      arg_shape.size());
                for (auto i = start_axis; i < target_shape.size(); i++)
                {
                    NODE_VALIDATION_CHECK(this,
                                          arg_shape[i - start_axis] == 1 ||
                                              arg_shape[i - start_axis] == target_shape[i],
                                          "Broadcast incorrect target shape. Expecting ",
                                          arg_shape[i - start_axis],
                                          " . Got ",
                                          target_shape[i]);
                }
            }
        }
    }

    set_input_is_relevant_to_shape(0); // arg - Result element type
    set_input_is_relevant_to_shape(1); // target_shape - Result shape
    set_input_is_relevant_to_shape(2); // axes_mapping - Broadcast type
    set_output_type(0, get_input_element_type(0), result_shape);
}

shared_ptr<Node> op::v1::Broadcast::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v1::Broadcast>(
        new_args.at(0), new_args.at(1), new_args.at(2), m_broadcast_spec);
}

void op::v1::Broadcast::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = input_value(0);

    auto broadcast_axes = get_broadcast_axes();
    if (broadcast_axes.first)
    {
        adjoints.add_delta(x, make_shared<op::Sum>(delta, broadcast_axes.second));
    }
    else
    {
        throw ngraph_error("Autodiff not supported on dynamic op variants");
    }
}

constexpr NodeTypeInfo op::v0::Broadcast::type_info;

op::v0::Broadcast::Broadcast(const OutputVector& args,
                             const Shape& shape,
                             const AxisSet& broadcast_axes)
    : Op(args)
    , m_shape(shape)
    , m_broadcast_axes(broadcast_axes)
{
    constructor_validate_and_infer_types();
}

op::v0::Broadcast::Broadcast(const Output<Node>& arg,
                             const Shape& shape,
                             const AxisSet& broadcast_axes)
    : Broadcast(OutputVector{arg}, shape, broadcast_axes)
{
}

bool op::v0::Broadcast::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("shape", m_shape);
    visitor.on_attribute("broadcast_axes", m_broadcast_axes);
    return true;
}

void op::v0::Broadcast::validate_and_infer_types()
{
    infer_shape();

    for (auto axis : m_broadcast_axes)
    {
        NODE_VALIDATION_CHECK(this,
                              axis < m_shape.size(),
                              "Broadcast axis index (",
                              axis,
                              ") exceeds specified output shape rank ",
                              "(broadcast axes: ",
                              m_broadcast_axes,
                              ", output shape: ",
                              m_shape,
                              ").");
    }

    Shape required_input_shape = m_shape;
    for (auto i = m_broadcast_axes.rbegin(); i != m_broadcast_axes.rend(); ++i)
    {
        required_input_shape.erase(required_input_shape.begin() + *i);
    }

    // TODO(amprocte): We can probably have a more helpful error message here.
    // There are two things that can go wrong, which are being picked up in
    // one fell swoop by this check: either the number of broadcast axes is not
    // enough, or there is a mismatch with one of the pre-broadcast axis lengths.
    NODE_VALIDATION_CHECK(
        this,
        get_input_partial_shape(0).compatible(required_input_shape),
        "Broadcast argument shape, specified output shape, and axes are incompatible ",
        "(argument shape: ",
        get_input_partial_shape(0),
        ", output shape: ",
        m_shape,
        ", broadcast axes: ",
        m_broadcast_axes,
        ").");

    set_output_type(0, get_input_element_type(0), m_shape);
}

shared_ptr<Node> op::v0::Broadcast::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v0::Broadcast>(new_args.at(0), m_shape, m_broadcast_axes);
}

void op::v0::Broadcast::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = input_value(0);

    adjoints.add_delta(x, make_shared<op::Sum>(delta, m_broadcast_axes));
}

constexpr NodeTypeInfo op::v0::BroadcastLike::type_info;

op::v0::BroadcastLike::BroadcastLike(const Output<Node>& arg,
                                     const Output<Node>& like_arg,
                                     const AxisSet& initial_broadcast_axes)
    : op::v0::Broadcast({arg, like_arg}, {}, {})
    , m_initial_broadcast_axes(initial_broadcast_axes)
{
    constructor_validate_and_infer_types();
}

bool op::v0::BroadcastLike::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("shape", m_shape);
    visitor.on_attribute("broadcast_axes", m_broadcast_axes);
    visitor.on_attribute("initial_broadcast_axes", m_initial_broadcast_axes);
    return true;
}

shared_ptr<Node> op::v0::BroadcastLike::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<v0::BroadcastLike>(new_args.at(0), new_args.at(1), m_initial_broadcast_axes);
}

void op::v0::BroadcastLike::infer_shape()
{
    const Shape& in_shape = get_input_shape(0);
    m_shape = get_input_shape(1);
    m_broadcast_axes = m_initial_broadcast_axes;
    if (m_broadcast_axes.size() == 0)
    {
        for (size_t i = 0; i < m_shape.size(); ++i)
        {
            if (i < in_shape.size())
            {
                if (in_shape.at(i) == 1 && m_shape.at(i) > 1)
                {
                    m_broadcast_axes.insert(i);
                }
            }
            else
            {
                m_broadcast_axes.insert(i);
            }
        }
    }
}
