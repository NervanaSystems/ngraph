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

#include "ngraph/log.hpp"
#include "ngraph/ops/convolution.hpp"

using namespace std;
using namespace ngraph;

op::Convolution::Convolution(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
    : RequiresTensorViewArgs("Convolution", {arg0, arg1})
{
    auto arg0_tensor_view_type = m_inputs.at(0).get_tensor_view_type();
    auto arg1_tensor_view_type = m_inputs.at(1).get_tensor_view_type();

    auto& arg0_element_type = arg0_tensor_view_type->get_element_type();
    auto& arg1_element_type = arg1_tensor_view_type->get_element_type();

    if (arg0_element_type == element::Bool::element_type() ||
        arg1_element_type == element::Bool::element_type())
    {
        throw ngraph_error("Arguments for convolution must have numeric element type");
    }

    if (arg0_element_type != arg1_element_type)
    {
        throw ngraph_error("Arguments must have the same element type");
    }

    auto arg0_shape = arg0_tensor_view_type->get_shape();
    auto arg1_shape = arg1_tensor_view_type->get_shape();

    if (arg0_shape.size() != arg1_shape.size())
    {
        throw ngraph_error("Arguments must have the same rank");
    }

    if (arg0_shape.size() < 2)
    {
        throw ngraph_error("Convolution arguments must have rank>=2");
    }

    if (arg0_shape.at(1) != arg1_shape.at(1))
    {
        throw ngraph_error("Number of input channels for convolution arguments does not match");
    }

    Shape out_shape;
    out_shape.push_back(arg0_shape[0]);
    out_shape.push_back(arg1_shape[0]);

    for (size_t i = 2; i < arg0_shape.size(); i++)
    {
        if (arg1_shape[i] == 0)
        {
            throw ngraph_error(
                "Convolution kernel must have size greater than 0 at each dimension");
        }

        if (arg1_shape[i] > arg0_shape[i])
        {
            throw ngraph_error(
                "Convolution kernel must be no larger than the image at each dimension");
        }

        out_shape.push_back(arg0_shape[i] - (arg1_shape[i] - 1));
    }

    set_value_type_checked(make_shared<TensorViewType>(arg0_element_type, out_shape));
}
