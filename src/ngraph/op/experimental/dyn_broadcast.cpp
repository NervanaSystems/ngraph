//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

using namespace std;
using namespace ngraph;

op::DynBroadcast::DynBroadcast(const shared_ptr<Node>& arg, const std::shared_ptr<Node>& shape_node, const std::shared_ptr<Node>& broadcast_axes_node)
    : Op("DynBroadcast", check_single_output_args({arg,shape_node,broadcast_axes_node}))
{
    constructor_validate_and_infer_types();
}

void op::DynBroadcast::validate_and_infer_types()
{
    NODE_VALIDATION_ASSERT(this, get_input_element_type(1).compatible(element::u64))
        << "Shape element type must be element::u64";
    NODE_VALIDATION_ASSERT(this, get_input_partial_shape(1).rank().compatible(1))
        << "Shape must be a vector";

    NODE_VALIDATION_ASSERT(this, get_input_element_type(2).compatible(element::u64))
        << "Broadcast axes element type must be element::u64";
    NODE_VALIDATION_ASSERT(this, get_input_partial_shape(2).rank().compatible(1))
        << "Broadcast axes must be a vector";

    // Without static value prop we won't know anything about the output shape.
    // (You might think we could at least know rank = rank(arg) + num_elements(broadcast_axes_node),
    // but broadcast_axes_node could contain duplicates and I think we want to allow that.)

    set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
}

shared_ptr<Node> op::DynBroadcast::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<DynBroadcast>(new_args.at(0), new_args.at(1), new_args.at(2));
}

void op::DynBroadcast::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    throw ngraph_error("DynBroadcast::generate_adjoints not yet implemented");
    // auto delta = deltas.at(0);
    // auto x = get_argument(0);
    // auto broadcast_axes = get_argument(2);
    // adjoints.add_delta(x, make_shared<op::DynSum>(delta, broadcast_axes));
}
