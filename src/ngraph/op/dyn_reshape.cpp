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

#include <algorithm>
#include <iostream>

#include "ngraph/function.hpp"
#include "ngraph/op/dyn_reshape.hpp"

using namespace std;
using namespace ngraph;

op::DynReshape::DynReshape(const shared_ptr<Node>& arg, const shared_ptr<Node>& shape)
    : Op("DynReshape", check_single_output_args({arg, shape}))
{
    constructor_validate_and_infer_types();
}

void op::DynReshape::validate_and_infer_types()
{
    /*Shape input_shape = get_input_shape(0);
    Shape output_shape = get_input_static_value(0);

    NODE_VALIDATION_ASSERT(this, output_shape.size() == 1)
        << "Output shape must have a rank of 1 (output shape: " << output_shape << ").";

    NODE_VALIDATION_ASSERT(this, shape_size(input_shape) == shape_size(output_shape))
        << "Number of elements in output shape does not match number of elements in argument shape "
        << "(output shape: " << output_shape << ", argument shape: " << input_shape << ").";

    set_output_type(0, get_input_element_type(0), output_shape);*/
}

shared_ptr<Node> op::DynReshape::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<DynReshape>(new_args.at(0), new_args.at(1));
}

void op::DynReshape::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    // TODO(amprocte): This will be DynReshape(delta,Shape(arg)).
    NGRAPH_FAIL() << "generate_adjoints for DynReshape not yet implemented";
}
