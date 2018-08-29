/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "ngraph/op/allreduce.hpp"

using namespace std;
using namespace ngraph;

op::AllReduce::AllReduce(const shared_ptr<Node>& arg)
    : RequiresTensorViewArgs("AllReduce", {arg})
{
    auto& input = m_inputs.at(0);
    set_value_type_checked(
        make_shared<TensorViewType>(input.get_element_type(), input.get_shape()));

    NODE_VALIDATION_ASSERT(
        this, arg->get_element_type() == element::f32 || arg->get_element_type() == element::f64)
        << "Only element types f32 and f64 are supported (argument element type: "
        << arg->get_element_type() << ")";
}

shared_ptr<Node> op::AllReduce::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args, 1);
    return make_shared<AllReduce>(new_args.at(0));
}
