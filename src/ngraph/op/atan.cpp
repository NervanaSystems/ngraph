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

#include "ngraph/op/atan.hpp"

#include "ngraph/axis_set.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/shape.hpp"

#include <string>
#include <vector>

using namespace std;
using namespace ngraph;

op::Atan::Atan(const shared_ptr<Node>& arg)
    : UnaryElementwiseArithmetic("Atan", arg)
{
}

shared_ptr<Node> op::Atan::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<Atan>(new_args.at(0));
}

void op::Atan::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = get_inputs().at(0).get_output().get_node();

    auto one = make_shared<op::Constant>(x->get_element_type(), Shape{}, vector<string>{"1"});

    AxisSet axes;
    for (size_t i = 0; i < x->get_shape().size(); i++)
        axes.insert(i);
    auto ones = make_shared<op::Broadcast>(one, x->get_shape(), axes);

    adjoints.add_delta(x, delta / (ones + x * x));
}
