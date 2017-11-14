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
#include "ngraph/ops/select.hpp"

using namespace std;
using namespace ngraph;

op::Select::Select(const std::shared_ptr<Node>& arg0,
                   const std::shared_ptr<Node>& arg1,
                   const std::shared_ptr<Node>& arg2)
    : RequiresTensorViewArgs("Select", Nodes{arg0, arg1, arg2})
{
    auto arg0_tensor_type = get_inputs().at(0).get_tensor_view_type();
    auto arg1_tensor_type = get_inputs().at(1).get_tensor_view_type();
    auto arg2_tensor_type = get_inputs().at(2).get_tensor_view_type();

    if (arg0_tensor_type->get_element_type() != element::Bool::element_type())
    {
        throw ngraph_error("Argument 0 for arithmetic operators must have boolean element type");
    }
    if (arg0_tensor_type->get_shape() != arg1_tensor_type->get_shape() ||
        arg0_tensor_type->get_shape() != arg2_tensor_type->get_shape())
    {
        throw ngraph_error("Arguments must have the same tensor view shape");
    }
    if (*arg1_tensor_type != *arg2_tensor_type)
    {
        throw ngraph_error("Arguments 1 and 2 must have the same tensor view type");
    }

    set_value_type_checked(arg1_tensor_type);
}
