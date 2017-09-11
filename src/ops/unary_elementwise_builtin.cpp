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

using namespace std;
using namespace ngraph;
using namespace ngraph::op;

void UnaryElementwiseBuiltin::propagate_types()
{
    if (m_arguments.size() != 1)
    {
        throw ngraph_error("Wrong number of arguments.");
    }

    auto arg_tensor_type =
        dynamic_pointer_cast<TensorViewType>(m_arguments.at(0)->get_value_type());
    if (nullptr == arg_tensor_type)
    {
        throw ngraph_error("Argument must be tensor view");
    }

    const element::Type& result_element_type =
        infer_result_element_type(arg_tensor_type->get_element_type());

    set_value_type_checked(make_shared<TensorViewType>(result_element_type,
                                                       arg_tensor_type->get_shape()));
}

/// Default behavior: result element type is same as argument element type.
const element::Type& UnaryElementwiseBuiltin::infer_result_element_type(
                         const element::Type& arg_element_type) const
{
    return arg_element_type;
}
