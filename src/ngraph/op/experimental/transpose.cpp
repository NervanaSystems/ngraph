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

#include <iostream>

#include "ngraph/op/experimental/transpose.hpp"

using namespace std;
using namespace ngraph;

op::Transpose::Transpose(const shared_ptr<Node>& arg, const shared_ptr<Node>& input_order)
    : Op("Transpose", check_single_output_args({arg, input_order}))
{
    constructor_validate_and_infer_types();
}

void op::Transpose::validate_and_infer_types()
{
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(1).compatible(element::i64),
                          "Input order must have element type i64.");

    auto& input_order_shape = get_input_partial_shape(1);
    NODE_VALIDATION_CHECK(
        this, input_order_shape.rank().compatible(1), "Input order must be a vector.");

    auto& arg_shape = get_input_partial_shape(0);
    NODE_VALIDATION_CHECK(this,
                          input_order_shape.compatible(PartialShape{arg_shape.rank()}),
                          "Input order must have shape [n], where n is the rank of arg.");

    set_output_type(0, get_input_element_type(0), PartialShape::dynamic(arg_shape.rank()));
}

shared_ptr<Node> op::Transpose::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Transpose>(new_args.at(0), new_args.at(1));
}

// TODO(amprocte): This will require some way of inverting the permutation in-graph. (TensorFlow,
// for example, has an InvertPermutation op, but that doesn't feel very nGraph-y somehow.)
void op::Transpose::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    throw ngraph_error("generate_adjoints not implemented for Transpose");
}
