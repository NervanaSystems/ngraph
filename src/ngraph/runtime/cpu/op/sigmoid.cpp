/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "ngraph/runtime/cpu/op/sigmoid.hpp"
#include "ngraph/log.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

shared_ptr<Node> op::Sigmoid::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<Sigmoid>(new_args.at(0));
}

op::Sigmoid::Sigmoid(shared_ptr<Node> input)
    : RequiresTensorViewArgs("Sigmoid", {input})
    , m_shape_input(input->get_shape())
{
    add_output(input->get_element_type(), m_shape_input);
}

op::SigmoidBackprop::SigmoidBackprop(shared_ptr<Node> arg, shared_ptr<Node> delta)
    : RequiresTensorViewArgs("SigmoidBackprop", {arg, delta})
{
    if (arg->get_element_type() != delta->get_element_type())
    {
        throw ngraph_error("Argument and delta element types for Sigmoid backprop do not match");
    }
    if (arg->get_shape() != delta->get_shape())
    {
        throw ngraph_error("Argument and delta shape for Sigmoid backprop do not match");
    }
    set_value_type_checked(delta->get_element_type(), delta->get_shape());
}

shared_ptr<Node> op::SigmoidBackprop::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<SigmoidBackprop>(new_args.at(0), new_args.at(1));
}

void op::Sigmoid::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto backprop = make_shared<op::SigmoidBackprop>(get_input_op(0), delta);
    adjoints.add_delta(get_input_op(0), backprop);
}
