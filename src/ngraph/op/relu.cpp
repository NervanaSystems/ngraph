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

#include "ngraph/op/relu.hpp"
#include "ngraph/op/multiply.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Relu::type_info;
constexpr NodeTypeInfo op::ReluBackprop::type_info;

op::Relu::Relu(const Output<Node>& arg)
    : UnaryElementwiseArithmetic(arg)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::Relu::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Relu>(new_args.at(0));
}

op::ReluBackprop::ReluBackprop(const Output<Node>& arg, const Output<Node>& delta)
    : BinaryElementwiseArithmetic(arg, delta, AutoBroadcastSpec::NONE)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::ReluBackprop::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<ReluBackprop>(new_args.at(0), new_args.at(1));
}

void op::Relu::generate_adjoints(autodiff::Adjoints& adjoints, const OutputVector& deltas)
{
    auto delta = deltas.at(0);

    auto backprop = make_shared<op::ReluBackprop>(output(0), delta);
    adjoints.add_delta(input_value(0), backprop);
}
