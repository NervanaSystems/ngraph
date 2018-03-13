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

#include "ngraph/runtime/cpu/ops/sigmoid.hpp"
#include "ngraph/log.hpp"
#include "ngraph/util.hpp"

std::shared_ptr<ngraph::Node>
    ngraph::op::Sigmoid::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return std::make_shared<Sigmoid>(new_args.at(0));
}

ngraph::op::Sigmoid::Sigmoid(std::shared_ptr<ngraph::Node> input)
    : RequiresTensorViewArgs("Sigmoid", {input})
    , m_shape_input(input->get_shape())
{
    add_output(input->get_element_type(), m_shape_input);
}

ngraph::op::SigmoidBackprop::SigmoidBackprop(std::shared_ptr<Node> arg, std::shared_ptr<Node> delta)
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

void ngraph::op::Sigmoid::generate_adjoints(ngraph::autodiff::Adjoints& adjoints,
                                            const std::shared_ptr<Node>& delta)
{
    auto backprop = std::make_shared<op::SigmoidBackprop>(get_input_op(0), delta);
    adjoints.add_delta(get_input_op(0), backprop);
}
