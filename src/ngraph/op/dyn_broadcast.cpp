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

#include "ngraph/op/dyn_broadcast.hpp"
#include "ngraph/op/sum.hpp"

using namespace std;
using namespace ngraph;

op::DynBroadcast::DynBroadcast(const shared_ptr<Node>& arg,
                         const shared_ptr<Node>& shape,
                         const shared_ptr<Node>& broadcast_axes)
    : Op("DynBroadcast", check_single_output_args({arg, shape, broadcast_axes}))
{
    constructor_validate_and_infer_types();
}

void op::DynBroadcast::validate_and_infer_types()
{ 
    // shape node should have integer data type. For now we only allow i64
    //TODO: potenially make the type more flexible to include other integer types
    auto shape_et = get_input_element_type(1);
    NODE_VALIDATION_ASSERT(this, shape_et==element::Type_t::i64)
        << "DynBroadcast shape has inocorect element type: " << shape_et << ", only i64 is allowed";

    //shape node should produce a one dimensional shape.    
    auto broadcast_shape_size = get_input_shape(1).size();
    NODE_VALIDATION_ASSERT(this, broadcast_shape_size==1)
        << "DynBroadcast shape has incorrect rank " << broadcast_shape_size << ", it must be equal to 1";

    // axes node should have integer data type. For now we only allow i64
    //TODO: potenially make the type more flexible to include other integer types
    auto axes_et = get_input_element_type(2);
    NODE_VALIDATION_ASSERT(this, axes_et==element::Type_t::i64)
        << "DynBroadcast axes input element type: " << axes_et << ", only i64 is allowed";

    //axes node should produce a one dimensional shape.    
    auto axes_shape_size = get_input_shape(2).size();
    NODE_VALIDATION_ASSERT(this, axes_shape_size==1)
        << "DynBroadcast axes has incorrect rank " << axes_shape_size << ", it must be equal to 1";

    set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
}

shared_ptr<Node> op::DynBroadcast::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<DynBroadcast>(new_args.at(0), new_args.at(1), new_args.at(2));
}

/// TODO: This function is not implemented!
void op::DynBroadcast::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = get_argument(0);

    // TODO adjoints.add_delta(x, make_shared<op::DynSum>(delta, m_broadcast_axes));
}

