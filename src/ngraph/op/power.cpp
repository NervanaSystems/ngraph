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

#include "ngraph/op/power.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/multiply.hpp"

using namespace std;
using namespace ngraph;

op::Power::Power(const shared_ptr<Node>& arg0, const shared_ptr<Node>& arg1)
    : BinaryElementwiseArithmetic("Power", arg0, arg1)
{
}

shared_ptr<Node> op::Power::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<Power>(new_args.at(0), new_args.at(1));
}

void op::Power::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = get_argument(0);
    auto y = get_argument(1);

    auto log_x = make_shared<op::Log>(x);

    adjoints.add_delta(x, delta * y * shared_from_this() / x);
    adjoints.add_delta(y, delta * shared_from_this() * log_x);
}
