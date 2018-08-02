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

#include "ngraph/runtime/cpu/op/dequantize.hpp"
#include "ngraph/op/constant.hpp"

ngraph::op::Dequantize::Dequantize(std::shared_ptr<Node> input,
                                   const float input_min_range,
                                   const float input_max_range,
                                   const element::Type& type)
    : RequiresTensorViewArgs("Dequantize", {input})
    , m_input_min(input_min_range)
    , m_input_max(input_max_range)
    , m_element_type(type)
{
    if (input_max_range < input_min_range)
    {
        throw ngraph_error("input max range should be greater than min range");
    }

    add_output(element::f32, input->get_shape());
}

std::shared_ptr<ngraph::Node>
    ngraph::op::Dequantize::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 4)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return std::make_shared<Dequantize>(new_args.at(0), m_input_min, m_input_max, m_element_type);
}
