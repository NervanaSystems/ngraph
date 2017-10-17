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

#include "ngraph/ops/sum.hpp"
#include "ngraph/function.hpp"

using namespace std;
using namespace ngraph::op;

void Sum::propagate_types()
{
    if (m_arguments.size() != 1)
    {
        throw ngraph_error("Wrong number of arguments.");
    }

    auto arg_type = m_arguments.at(0)->get_value_type();
    if (nullptr == arg_type)
    {
        throw ngraph_error("Argument to sum is missing type.");
    }
    auto arg_tensor_view_type = dynamic_pointer_cast<const TensorViewType>(arg_type);
    if (nullptr == arg_tensor_view_type)
    {
        throw ngraph_error("Argument to sum is not a tensor view");
    }

    auto& arg_element_type = arg_tensor_view_type->get_element_type();
    if (arg_element_type == element::Bool::element_type())
    {
        throw ngraph_error("Argument for sum must have numeric element type");
    }

    auto arg_shape = arg_tensor_view_type->get_shape();

    for (auto axis : m_reduction_axes)
    {
        if (axis >= arg_shape.size())
        {
            throw ngraph_error("Reduction axis for sum is out of bounds");
        }
    }

    Shape result_shape;

    for (size_t i = 0; i < arg_shape.size(); i++)
    {
        if (m_reduction_axes.count(i) == 0)
        {
            result_shape.push_back(arg_shape.at(i));
        }
    }

    set_value_type_checked(
        make_shared<TensorViewType>(arg_tensor_view_type->get_element_type(), result_shape));
}
