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

#include "ngraph/op/sigmoid.hpp"
#include "ngraph/log.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

shared_ptr<Node> op::Sigmoid::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args, 1);
    return make_shared<Sigmoid>(new_args.at(0));
}

op::Sigmoid::Sigmoid(shared_ptr<Node> arg)
    : UnaryElementwiseArithmetic("Sigmoid", {arg})
{
    set_output_type(0, arg->get_element_type(), arg->get_shape());
    constructor_validate_and_infer_types();
}

op::SigmoidBackprop::SigmoidBackprop(shared_ptr<Node> arg, shared_ptr<Node> delta)
    : Op("SigmoidBackprop", check_single_output_args({arg, delta}))
{
    NODE_VALIDATION_ASSERT(this, arg->get_element_type() == delta->get_element_type())
        << "Argument and delta element types do not match (argument element type: "
        << arg->get_element_type() << ", delta element type: " << delta->get_element_type() << ").";

    NODE_VALIDATION_ASSERT(this, arg->get_shape() == delta->get_shape())
        << "Argument and delta shapes do not match (argument shape: " << arg->get_shape()
        << ", delta shape: " << delta->get_shape() << ").";

    set_output_type(0, delta->get_element_type(), delta->get_shape());
}

shared_ptr<Node> op::SigmoidBackprop::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args, 2);
    return make_shared<SigmoidBackprop>(new_args.at(0), new_args.at(1));
}

void op::Sigmoid::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto backprop = make_shared<op::SigmoidBackprop>(get_argument(0), delta);
    adjoints.add_delta(get_argument(0), backprop);
}
