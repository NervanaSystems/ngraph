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

#include <memory>
#include <typeindex>
#include <typeinfo>
#include "ngraph/node.hpp"

#include "ngraph/op/result.hpp"

using namespace std;
using namespace ngraph;

op::Result::Result(const shared_ptr<Node>& arg)
    : RequiresTensorViewArgs("Result", {arg})
{
    NODE_VALIDATION_ASSERT(this, arg->get_outputs().size() == 1)
        << "Argument has " << arg->get_outputs().size() << " outputs (1 expected).";

    // always borrow the placement conf even the default one
    set_placement(arg->get_placement());
    set_value_type_checked(arg->get_element_type(), arg->get_shape());
}

shared_ptr<Node> op::Result::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args, 1);

    NODE_VALIDATION_ASSERT(this, new_args[0]->get_outputs().size() == 1)
        << "Argument has " << new_args[0]->get_outputs().size() << " outputs (1 expected).";

    auto res = make_shared<Result>(new_args.at(0));
    res->set_needs_copy(m_needs_copy);
    res->set_needs_default_layout(m_needs_default_layout);
    return res;
}

void op::Result::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    adjoints.add_delta(get_argument(0), delta);
}
