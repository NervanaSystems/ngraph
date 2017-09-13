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

void BinaryElementwiseBuiltin::propagate_types()
{
    if (m_arguments.size() != 2)
    {
        throw ngraph_error("Wrong number of arguments.");
    }

    auto arg0_tensor_type =
        dynamic_pointer_cast<TensorViewType>(m_arguments.at(0)->get_value_type());
    auto arg1_tensor_type =
        dynamic_pointer_cast<TensorViewType>(m_arguments.at(1)->get_value_type());
    if (nullptr == arg0_tensor_type || nullptr == arg1_tensor_type)
    {
        throw ngraph_error("Arguments must be tensor views");
    }
    if (arg0_tensor_type->get_shape() != arg1_tensor_type->get_shape())
    {
        throw ngraph_error("Arguments must have the same tensor view shape");
    }

    const element::Type& result_element_type =
        infer_result_element_type(arg0_tensor_type->get_element_type(),
                                  arg1_tensor_type->get_element_type());

    set_value_type_checked(make_shared<TensorViewType>(result_element_type,
                                                       arg0_tensor_type->get_shape()));
}

/// TODO: make this abstract
/// 
/// Default behavior: arg0 and arg1 element types must match, and result
/// element type is same as argument element types.
const element::Type& BinaryElementwiseBuiltin::infer_result_element_type(
                         const element::Type& arg0_element_type,
                         const element::Type& arg1_element_type) const
{
    if (arg0_element_type != arg1_element_type)
    {
        throw ngraph_error("Arguments must have the same tensor view element type");
    }

    return arg0_element_type;
}
