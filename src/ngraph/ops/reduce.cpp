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

#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph::op;

void Reduce::propagate_types()
{
    // TODO: For now we have to assume the reduction function is correctly typed.

    if (m_arguments.size() != 2)
    {
        throw ngraph_error("Wrong number of arguments.");
    }

    auto arg_reductee_type = m_arguments.at(0)->get_value_type();
    if (nullptr == arg_reductee_type)
    {
        throw ngraph_error("Argument to reduce is missing type.");
    }
    auto arg_reductee_tensor_view_type = dynamic_pointer_cast<const TensorViewType>(arg_reductee_type);
    if (nullptr == arg_reductee_tensor_view_type)
    {
        throw ngraph_error("Argument to reduce is not a tensor view");
    }

    auto arg_init_type = m_arguments.at(1)->get_value_type();
    if (nullptr == arg_init_type)
    {
        throw ngraph_error("Argument for initial value is missing type.");
    }
    auto arg_init_tensor_view_type = dynamic_pointer_cast<const TensorViewType>(arg_init_type);
    if (nullptr == arg_init_tensor_view_type)
    {
        throw ngraph_error("Argument for initial value is not a tensor view");
    }
    if (arg_init_tensor_view_type->get_shape().size() != 0)
    {
        throw ngraph_error("Argument for initial value is not a scalar");
    }

    if (arg_init_tensor_view_type->get_element_type() != arg_reductee_tensor_view_type->get_element_type())
    {
        throw ngraph_error("Element types for reductee and initial values do not match");
    }

    auto arg_reductee_shape = arg_reductee_tensor_view_type->get_shape();

    for (auto axis : m_reduction_axes)
    {
        if (axis >= arg_reductee_shape.size())
        {
            throw ngraph_error("Reduction axis is out of bounds");
        }
    }

    vector<size_t> result_shape = {};

    for (auto i = 0; i < arg_reductee_shape.size(); i++)
    {
        if (m_reduction_axes.count(i) == 0)
        {
            result_shape.push_back(arg_reductee_shape.at(i));
        }
    }

    set_value_type_checked(make_shared<TensorViewType>(arg_reductee_tensor_view_type->get_element_type(), result_shape));
}
