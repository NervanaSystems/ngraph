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
#include "ngraph/ops/convert.hpp"
#include "ngraph/ops/multiply.hpp"
#include "ngraph/ops/not.hpp"
#include "ngraph/ops/select.hpp"

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

void ngraph::op::Select::generate_adjoints(autodiff::Adjoints& adjoints,
                                           const std::shared_ptr<Node>& delta)
{
    auto p = m_arguments[0];
    auto x = m_arguments[1];
    auto y = m_arguments[2];

    auto p_as_float = std::make_shared<op::Convert>(p,element::Float32::element_type());
    auto not_p_as_float = std::make_shared<op::Convert>(std::make_shared<op::Not>(p),element::Float32::element_type());

    adjoints.add_delta(x, delta * p_as_float);
    adjoints.add_delta(y, delta * not_p_as_float);
}
