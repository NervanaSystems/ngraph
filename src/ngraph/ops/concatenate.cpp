// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <memory>

#include "ngraph/ops/concatenate.hpp"

using namespace std;
using namespace ngraph;

op::Concat::Concat(const Nodes& args, size_t concatenation_axis)
    : RequiresTensorViewArgs("Concat", args)
    , m_concatenation_axis(concatenation_axis)
{
    if (get_arguments().size() < 1) //TODO: [nikolayk] fix in the next iteration
    {
        throw ngraph_error("At least one argument required");
    }

    auto arg0_tensor_view_type = get_inputs().at(0).get_tensor_view_type();
    auto arg0_shape = arg0_tensor_view_type->get_shape();
    if (m_concatenation_axis >= arg0_shape.size())
    {
        throw ngraph_error("Concatenation axis is out of bounds");
    }

    size_t concatenation_axis_length = arg0_shape.at(m_concatenation_axis);
    auto& arg0_element_type = arg0_tensor_view_type->get_element_type();

    for (auto i = 1; i < get_inputs().size(); i++)
    {
        auto argi_tensor_view_type = get_inputs().at(i).get_tensor_view_type();
        auto argi_shape = argi_tensor_view_type->get_shape();
        if (argi_shape.size() != arg0_shape.size())
        {
            throw ngraph_error("Arguments to concat do not have same rank");
        }

        if (argi_tensor_view_type->get_element_type() != arg0_element_type)
        {
            throw ngraph_error("Argument element types do not match");
        }

        for (auto j = 0; j < argi_shape.size(); j++)
        {
            if (j != m_concatenation_axis && arg0_shape.at(j) != argi_shape.at(j))
            {
                throw ngraph_error(
                    "Arguments to concat do not have same dimension on a non-concatenation axis");
            }
            else if (j == m_concatenation_axis)
            {
                concatenation_axis_length += argi_shape.at(j);
            }
        }
    }
    vector<size_t> concatenated_shape = arg0_shape;
    concatenated_shape.at(m_concatenation_axis) = concatenation_axis_length;

    set_value_type_checked(make_shared<TensorViewType>(arg0_element_type, concatenated_shape));
}
