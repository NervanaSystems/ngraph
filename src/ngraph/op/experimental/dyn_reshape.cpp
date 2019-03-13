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

#include <algorithm>
#include <iostream>

#include "ngraph/function.hpp"
#include "ngraph/op/experimental/dyn_reshape.hpp"

using namespace std;
using namespace ngraph;

op::DynReshape::DynReshape(const shared_ptr<Node>& arg,
                           const shared_ptr<Node>& output_shape)
    : Op("DynReshape", check_single_output_args({arg, output_shape}))
{
    constructor_validate_and_infer_types();
}

void op::DynReshape::validate_and_infer_types()
{
    auto output_shape_et = get_input_element_type(1);
    // check data types
    NODE_VALIDATION_CHECK(this,
                          output_shape_et == element::Type_t::i64,
                          "output_shape element type should be i64.");
    
    // check shapes
    auto output_shape = get_input_partial_shape(1); 
    NODE_VALIDATION_CHECK(this,
                          output_shape.rank().compatible(1),
                          "output_shape should have rank 1, got ", output_shape.rank(), ".");
    
    set_output_type(0, get_input_element_type(0), output_shape);
}

shared_ptr<Node> op::DynReshape::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<DynReshape>(new_args.at(0), new_args.at(1));
}

void op::DynReshape::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    throw ngraph_error("generate_adjoints not implemented for DynReshape");
}
