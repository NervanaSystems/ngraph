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

#include <memory>

#include "ngraph/op/convert.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

op::Maximum::Maximum(const shared_ptr<Node>& arg0, const shared_ptr<Node>& arg1)
    : BinaryElementwiseArithmetic("Maximum", arg0, arg1)
{
}

shared_ptr<Node> op::Maximum::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<Maximum>(new_args.at(0), new_args.at(1));
}

void op::Maximum::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = get_input_op(0);
    auto y = get_input_op(1);
    adjoints.add_delta(
        x, delta * make_shared<op::Convert>(make_shared<op::Greater>(x, y), x->get_element_type()));
    adjoints.add_delta(
        y, delta * make_shared<op::Convert>(make_shared<op::Greater>(y, x), y->get_element_type()));
}
