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

using namespace std;
using namespace ngraph;

op::Convert::Convert(const shared_ptr<Node>& arg, const element::Type& element_type)
    : UnaryElementwise("Convert", element_type, arg)
    , m_element_type(element_type)
{
}

shared_ptr<Node> op::Convert::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<Convert>(new_args.at(0), m_element_type);
}

void op::Convert::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = get_argument(0);

    adjoints.add_delta(x, make_shared<op::Convert>(delta, x->get_element_type()));
}
