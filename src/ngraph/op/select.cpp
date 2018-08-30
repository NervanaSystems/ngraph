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

#include "ngraph/log.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/not.hpp"
#include "ngraph/op/select.hpp"

using namespace std;
using namespace ngraph;

op::Select::Select(const shared_ptr<Node>& arg0,
                   const shared_ptr<Node>& arg1,
                   const shared_ptr<Node>& arg2)
    : RequiresTensorViewArgs("Select", NodeVector{arg0, arg1, arg2})
{
    auto& input_0 = get_inputs().at(0);
    auto& input_1 = get_inputs().at(1);
    auto& input_2 = get_inputs().at(2);

    if (input_0.get_element_type() != element::boolean)
    {
        throw ngraph_error("Argument 0 for arithmetic operators must have boolean element type");
    }
    if (input_0.get_shape() != input_1.get_shape() || input_0.get_shape() != input_2.get_shape())
    {
        throw ngraph_error("Arguments must have the same shape");
    }
    if (input_1.get_element_type() != input_2.get_element_type())
    {
        throw ngraph_error("Arguments 1 and 2 must have the same element type");
    }

    set_value_type_checked(input_1.get_element_type(), input_1.get_shape());
}

shared_ptr<Node> op::Select::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 3)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<Select>(new_args.at(0), new_args.at(1), new_args.at(2));
}

void op::Select::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto p = get_inputs().at(0).get_output().get_node();
    auto x = get_inputs().at(1).get_output().get_node();
    auto y = get_inputs().at(2).get_output().get_node();

    auto p_as_x_type = make_shared<op::Convert>(p, x->get_element_type());
    auto not_p_as_y_type = make_shared<op::Convert>(make_shared<op::Not>(p), y->get_element_type());

    adjoints.add_delta(x, delta * p_as_x_type);
    adjoints.add_delta(y, delta * not_p_as_y_type);
}
