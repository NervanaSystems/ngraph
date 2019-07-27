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

#include "ngraph/op/experimental/dyn_broadcast.hpp"
#include "ngraph/op/sum.hpp"

using namespace std;
using namespace ngraph;

const string op::DynBroadcast::type_name{"DynBroadcast"};

op::DynBroadcast::DynBroadcast(const shared_ptr<Node>& arg,
                               const shared_ptr<Node>& shape,
                               const shared_ptr<Node>& broadcast_axes)
    : Op(check_single_output_args({arg, shape, broadcast_axes}))
{
    constructor_validate_and_infer_types();
}

void op::DynBroadcast::validate_and_infer_types()
{
    // shape node should have integer data type. For now we only allow i64
    //TODO: potenially make the type more flexible to include other integer types
    auto shape_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          shape_et.compatible(element::Type_t::i64),
                          "DynBroadcast shape must have element type i64, but has ",
                          shape_et);

    //shape node should produce a one dimensional shape.
    auto broadcast_shape_rank = get_input_partial_shape(1).rank();
    NODE_VALIDATION_CHECK(this,
                          broadcast_shape_rank.compatible(1),
                          "DynBroadcast shape rank must be 1, but has ",
                          broadcast_shape_rank);

    // axes node should have integer data type. For now we only allow i64
    //TODO: potenially make the type more flexible to include other integer types
    auto axes_et = get_input_element_type(2);
    NODE_VALIDATION_CHECK(this,
                          axes_et.compatible(element::Type_t::i64),
                          "DynBroadcast axes must have element type i64, but has ",
                          axes_et);

    //axes node should produce a one dimensional shape.
    auto axes_shape_rank = get_input_partial_shape(2).rank();
    NODE_VALIDATION_CHECK(this,
                          axes_shape_rank.compatible(1),
                          "DynBroadcast axes rank must be 1, but has ",
                          axes_shape_rank);

    set_input_is_relevant_to_shape(1);
    set_input_is_relevant_to_shape(2);
    set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
}

shared_ptr<Node> op::DynBroadcast::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<DynBroadcast>(new_args.at(0), new_args.at(1), new_args.at(2));
}

// TODO: This function is not implemented!
void op::DynBroadcast::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    throw ngraph_error("generate_adjoints not implemented for DynBroadcast");
}
