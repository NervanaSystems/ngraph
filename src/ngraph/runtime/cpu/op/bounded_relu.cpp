//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#include "ngraph/runtime/cpu/op/bounded_relu.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

op::BoundedRelu::BoundedRelu(shared_ptr<Node> arg, float alpha)
    : RequiresTensorViewArgs("BoundedRelu", {arg})
    , m_alpha(alpha)
{
    set_value_type_checked(arg->get_element_type(), arg->get_shape());
}

shared_ptr<Node> op::BoundedRelu::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<BoundedRelu>(new_args.at(0), m_alpha);
}
