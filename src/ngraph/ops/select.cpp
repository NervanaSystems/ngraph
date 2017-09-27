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

#include "ngraph/ngraph.hpp"
#include "ngraph/log.hpp"

using namespace std;
using namespace ngraph;
using namespace ngraph::op;

void Select::propagate_types()
{
    if (m_arguments.size() != 3)
    {
        throw ngraph_error("Wrong number of arguments.");
    }

    auto arg0_tensor_type =
        dynamic_pointer_cast<const TensorViewType>(m_arguments.at(0)->get_value_type());
    auto arg1_tensor_type =
        dynamic_pointer_cast<const TensorViewType>(m_arguments.at(1)->get_value_type());
    auto arg2_tensor_type =
        dynamic_pointer_cast<const TensorViewType>(m_arguments.at(2)->get_value_type());
    if (nullptr == arg0_tensor_type || nullptr == arg1_tensor_type || nullptr == arg2_tensor_type)
    {
        throw ngraph_error("Arguments must be tensor views");
    }
    if (arg0_tensor_type->get_element_type() != element::Bool::element_type())
    {
        throw ngraph_error("Argument 0 for arithmetic operators must have boolean element type");
    }
    if (arg0_tensor_type->get_shape() != arg1_tensor_type->get_shape()
     || arg0_tensor_type->get_shape() != arg2_tensor_type->get_shape())
    {
        throw ngraph_error("Arguments must have the same tensor view shape");
    }
    if (*arg1_tensor_type != *arg2_tensor_type)
    {
        throw ngraph_error("Arguments 1 and 2 must have the same tensor view type");
    }

    set_value_type_checked(arg1_tensor_type);
}

