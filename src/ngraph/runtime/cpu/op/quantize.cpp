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

#include "ngraph/runtime/cpu/op/quantize.hpp"
#include "ngraph/op/constant.hpp"

ngraph::op::Quantize::Quantize(std::shared_ptr<Node> input,
                               std::shared_ptr<Node> min,
                               std::shared_ptr<Node> max,
                               const element::Type& type)
    : Op("Quantize", check_single_output_args({input, min, max}))
    , m_element_type(type)
{
    constructor_validate_and_infer_types();

    if (input->get_element_type() != element::f32)
    {
        throw ngraph_error("Quantization supported only from float32 --> i8/u8!");
    }

    if (min->get_element_type() != min->get_element_type())
    {
        throw ngraph_error("Min's element type isn't equal to max's!");
    }

    if (min->get_shape().size() != 0)
    {
        throw ngraph_error("Min is not a scalar!");
    }

    if (max->get_shape().size() != 0)
    {
        throw ngraph_error("Max is not a scalar!");
    }

    if (!(std::dynamic_pointer_cast<op::Constant>(min) &&
          std::dynamic_pointer_cast<op::Constant>(max)))
    {
        throw ngraph_error("Min and max have to be constants!");
    }

    auto min_const_op = std::static_pointer_cast<ngraph::op::Constant>(min);
    auto max_const_op = std::static_pointer_cast<ngraph::op::Constant>(max);
    float input_min_range = *(static_cast<float const*>(min_const_op->get_data_ptr()));
    float input_max_range = *(static_cast<float const*>(max_const_op->get_data_ptr()));
    this->m_input_min = input_min_range;
    this->m_input_max = input_max_range;

    set_output_size(3);
    set_output_type(0, type, input->get_shape());
    set_output_type(1, element::f32, Shape{});
    set_output_type(2, element::f32, Shape{});
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
