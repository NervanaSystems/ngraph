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
#include "ngraph/op/sum.hpp"
#include "ngraph/shape_util.hpp"

using namespace std;
using namespace ngraph;

op::Broadcast::Broadcast(const shared_ptr<Node>& arg,
                         const shared_ptr<Node>& broadcast_shape,
                         const shared_ptr<Node>& broadcast_axes)
    : Op("Broadcast", check_single_output_args({arg,broadcast_shape,broadcast_axes}))
{
    constructor_validate_and_infer_types();
}

op::Broadcast::Broadcast(const shared_ptr<Node>& arg,
                         const Shape& shape,
                         const AxisSet& broadcast_axes)
    : Broadcast(arg, shape_to_i64_constant(shape), axis_set_to_i64_constant(broadcast_axes))
{
}

void op::Broadcast::validate_and_infer_types()
{
    auto broadcast_shape_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          broadcast_shape_et.compatible(element::Type_t::i64),
                          "Broadcast shape (input 1) must have element type i64");

    auto broadcast_shape_rank = get_input_partial_shape(1).rank();
    NODE_VALIDATION_CHECK(this,
                          broadcast_shape_rank.compatible(1),
                          "Broadcast shape (input 1) must be a vector.");

    auto broadcast_axes_et = get_input_element_type(2);
    NODE_VALIDATION_CHECK(this,
                          broadcast_axes_et.compatible(element::Type_t::i64),
                          "Broadcast axes (input 2) must have element type i64.");

    auto broadcast_axes_rank = get_input_partial_shape(2).rank();
    NODE_VALIDATION_CHECK(this,
                          broadcast_axes_rank.compatible(1),
                          "Broadcast axes (input 2) must be a vector.");

    if (broadcast_shape_is_constant() && broadcast_axes_are_constant())
    {
        Shape broadcast_shape = get_broadcast_shape();
        AxisSet broadcast_axes = get_broadcast_axes();

        for (auto axis : broadcast_axes)
        {
            NODE_VALIDATION_CHECK(this,
                                axis < broadcast_shape.size(),
                                "Broadcast axis index (",
                                axis,
                                ") exceeds specified output shape rank ",
                                "(broadcast axes: ",
                                broadcast_axes,
                                ", output shape: ",
                                broadcast_shape,
                                ").");
        }

        Shape required_input_shape = broadcast_shape;
        for (auto i = broadcast_axes.rbegin(); i != broadcast_axes.rend(); ++i)
        {
            required_input_shape.erase(required_input_shape.begin() + *i);
        }

        // TODO(amprocte): We can probably have a more helpful error message here.
        // There are two things that can go wrong, which are being picked up in
        // one fell swoop by this check: either the number of broadcast axes is not
        // enough, or there is a mismatch with one of the pre-broadcast axis lengths.
        //
        // TODO(amprocte): There is a temporary(?) hack here that allows the case
        // where input shape is Shape{} and the axis set is empty. I'm putting this
        // here to make it possible to get rid of BroadcastLike, but don't know if
        // we want to keep it. (There will be a way to do BroadcastLike without this
        // once the Range op is in.)
        NODE_VALIDATION_CHECK(
            this,
            (get_input_partial_shape(0).rank().compatible(0) && broadcast_axes.empty()) || get_input_partial_shape(0).compatible(required_input_shape),
            "Broadcast argument shape, specified output shape, and axes are incompatible ",
            "(argument shape: ",
            get_input_partial_shape(0),
            ", output shape: ",
            broadcast_shape,
            ", broadcast axes: ",
            broadcast_axes,
            ").");

        set_output_type(0, get_input_element_type(0), broadcast_shape);
    }
    else
    {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
    }

    set_input_is_relevant_to_shape(1, true);
    set_input_is_relevant_to_shape(2, true);
}

shared_ptr<Node> op::Broadcast::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Broadcast>(new_args.at(0), new_args.at(1), new_args.at(2));
}

void op::Broadcast::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = get_argument(0);

    adjoints.add_delta(x, make_shared<op::Sum>(delta, get_broadcast_axes()));
}

Shape op::Broadcast::get_broadcast_shape() const
{
    NGRAPH_ASSERT(broadcast_shape_is_constant());
    return shape_from_i64_constant(get_argument(1));
}

bool op::Broadcast::broadcast_shape_is_constant() const
{
    return get_argument(1)->is_constant();
}

AxisSet op::Broadcast::get_broadcast_axes() const
{
    NGRAPH_ASSERT(broadcast_axes_are_constant());
    return axis_set_from_i64_constant(get_argument(2));
}

bool op::Broadcast::broadcast_axes_are_constant() const
{
    return get_argument(2)->is_constant();
}
