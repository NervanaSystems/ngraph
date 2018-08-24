
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

#include "ngraph/runtime/cpu/op/quantize.hpp"

ngraph::op::Quantize::Quantize(std::shared_ptr<Node> input,
                               std::shared_ptr<Node> min,
                               std::shared_ptr<Node> max,
                               const element::Type& type)
    : RequiresTensorViewArgs("Quantize", {input, min, max})
    , m_element_type(type)
{
    add_output(type, input->get_shape());
    add_output(element::f32, Shape{1});
    add_output(element::f32, Shape{1});
}

std::shared_ptr<ngraph::Node>
    ngraph::op::Quantize::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 3)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return std::make_shared<Quantize>(
        new_args.at(0), new_args.at(1), new_args.at(2), m_element_type);
}
