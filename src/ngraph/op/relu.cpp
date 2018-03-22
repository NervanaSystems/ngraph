/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include "ngraph/op/relu.hpp"
#include "ngraph/op/multiply.hpp"

using namespace std;
using namespace ngraph;

op::Relu::Relu(shared_ptr<Node> arg)
    : UnaryElementwiseArithmetic("Relu", {arg})
{
    set_value_type_checked(arg->get_element_type(), arg->get_shape());
}

op::ReluBackprop::ReluBackprop(shared_ptr<Node> arg, shared_ptr<Node> delta)
    : RequiresTensorViewArgs("ReluBackprop", {arg, delta})
{
    if (arg->get_element_type() != delta->get_element_type())
    {
        throw ngraph_error("Argument and delta element types for Relu backprop do not match");
    }
    if (arg->get_shape() != delta->get_shape())
    {
        throw ngraph_error("Argument and delta shape for Relu backprop do not match");
    }
    set_value_type_checked(delta->get_element_type(), delta->get_shape());
}

void op::Relu::generate_adjoints(autodiff::Adjoints& adjoints, const std::shared_ptr<Node>& delta)
{
    auto backprop = std::make_shared<op::ReluBackprop>(get_input_op(0), delta);
    adjoints.add_delta(get_input_op(0), backprop);
}
