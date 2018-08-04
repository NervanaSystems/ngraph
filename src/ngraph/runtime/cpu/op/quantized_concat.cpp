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

#include <cassert>
#include <memory>

#include "ngraph/op/concat.hpp"
#include "ngraph/op/slice.hpp"
#include "quantized_concat.hpp"

using namespace std;
using namespace ngraph;

op::QuantizedConcat::QuantizedConcat(const NodeVector& args,
                                     size_t concatenation_axis,
                                     vector<float>& input_mins,
                                     vector<float>& input_maxes)
    : RequiresTensorViewArgs("QuantizedConcat", args)
    , m_concatenation_axis(concatenation_axis)
    , m_input_mins(input_mins)
    , m_input_maxes(input_maxes)
{
    if (m_inputs.size() < 1)
    {
        throw ngraph_error("At least one argument required");
    }

    auto& input_0 = get_inputs().at(0);
    auto input_0_shape = input_0.get_shape();
    if (m_concatenation_axis >= input_0_shape.size())
    {
        throw ngraph_error("QuantizedConcatenation axis is out of bounds");
    }

    size_t concatenation_axis_length = input_0_shape.at(m_concatenation_axis);
    auto& input_0_element_type = input_0.get_element_type();

    for (auto i = 1; i < get_inputs().size(); i++)
    {
        auto& input_i = get_inputs().at(i);
        auto input_i_shape = input_i.get_shape();
        if (input_i_shape.size() != input_0_shape.size())
        {
            throw ngraph_error("Arguments to concat do not have same rank");
        }

        if (input_i.get_element_type() != input_0_element_type)
        {
            throw ngraph_error("Argument element types do not match");
        }

        for (auto j = 0; j < input_i_shape.size(); j++)
        {
            if (j != m_concatenation_axis && input_0_shape.at(j) != input_i_shape.at(j))
            {
                throw ngraph_error(
                    "Arguments to concat do not have same dimension on a non-concatenation axis");
            }
            else if (j == m_concatenation_axis)
            {
                concatenation_axis_length += input_i_shape.at(j);
            }
        }
    }
    vector<size_t> concatenated_shape = input_0_shape;
    concatenated_shape.at(m_concatenation_axis) = concatenation_axis_length;

    set_value_type_checked(make_shared<TensorViewType>(input_0_element_type, concatenated_shape));
    add_output(element::f32, Shape{1});
    add_output(element::f32, Shape{1});
}

shared_ptr<Node> op::QuantizedConcat::copy_with_new_args(const NodeVector& new_args) const
{
    return make_shared<QuantizedConcat>(
        new_args, m_concatenation_axis, m_input_mins, m_input_maxes);
}
