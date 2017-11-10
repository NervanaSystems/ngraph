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

#include "ngraph/ops/op.hpp"

using namespace std;
using namespace ngraph;

op::UnaryElementwise::UnaryElementwise(
    std::function<const element::Type&(const element::Type&)> element_type_function,
    const std::shared_ptr<Node>& arg)
    : RequiresTensorViewArgs(Nodes{arg})
{
    auto arg_tensor_type = get_inputs().at(0).get_tensor_view_type();
    const element::Type& result_element_type =
        element_type_function(arg_tensor_type->get_element_type());

    set_value_type_checked(
        make_shared<TensorViewType>(result_element_type, arg_tensor_type->get_shape()));
}
