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

#include "ngraph.hpp"

using namespace std;
using namespace ngraph::op;

void Dot::propagate_types()
{
    auto arg0_tensor_type =
        dynamic_pointer_cast<TensorViewType>(m_arguments.at(0)->get_value_type());
    auto arg1_tensor_type =
        dynamic_pointer_cast<TensorViewType>(m_arguments.at(1)->get_value_type());
    if (nullptr == arg0_tensor_type || nullptr == arg1_tensor_type)
    {
        throw ngraph_error("Arguments to dot must be tensor views");
    }
    if (arg0_tensor_type->get_element_type() != arg1_tensor_type->get_element_type())
    {
        throw ngraph_error("Arguments to dot must have the same element type");
    }

    vector<size_t> arg0_shape     = arg0_tensor_type->get_shape();
    vector<size_t> arg1_shape     = arg1_tensor_type->get_shape();
    size_t         arg0_reduction = arg0_shape.size() - 1;
    size_t         arg1_reduction;
    const bool     is_scalar_mult = arg0_shape.size() == 0 || arg1_shape.size() == 0;

    if (arg1_shape.size() > 1)
    {
        arg1_reduction = arg1_shape.size() - 2;
    }
    else
    {
        arg1_reduction = arg1_shape.size() - 1;
    }
    if (!is_scalar_mult && (arg0_shape.at(arg0_reduction) != arg1_shape.at(arg1_reduction)))
    {
        throw ngraph_error("Dot reduction axes not compatible");
    }

    vector<size_t> result_shape;
    result_shape.reserve(arg0_shape.size() + arg1_shape.size() - (is_scalar_mult ? 0 : 2));

    for(auto i = 0; i < arg0_shape.size(); i++)
    {
        if(is_scalar_mult || i != arg0_reduction)
        {
            result_shape.push_back(arg0_shape[i]);
        }
    }

    for(auto i = 0; i < arg1_shape.size(); i++)
    {
        if(is_scalar_mult || i != arg1_reduction)
        {
            result_shape.push_back(arg1_shape[i]);
        }
    }

    auto result_type = make_shared<TensorViewType>(arg0_tensor_type->get_element_type(), result_shape);
    set_value_type_checked(result_type);
}
