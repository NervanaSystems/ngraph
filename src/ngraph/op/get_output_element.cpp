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

#include <sstream>

#include "ngraph/op/get_output_element.hpp"

using namespace std;
using namespace ngraph;

op::GetOutputElement::GetOutputElement(const shared_ptr<Node>& arg, size_t n)
    : Node("GetOutputElement", {arg})
    , m_n{n}
{
    if (m_n >= arg->get_output_size())
    {
        throw ngraph_error("Indexing tuple beyond its size");
    }

    set_value_type_checked(arg->get_output_element_type(n), arg->get_output_shape(n));
}

shared_ptr<Node> op::GetOutputElement::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<GetOutputElement>(new_args.at(0), m_n);
}

NodeVector op::GetOutputElement::get_input_ops()
{
    return NodeVector{get_inputs().at(0).get_output().get_node()};
}

void op::GetOutputElement::generate_adjoints(autodiff::Adjoints& adjoints,
                                             const shared_ptr<Node>& delta)
{
    //Filter out updates(deltas) from mean and variance (for batchnorm)
    //as dinput is the only update required.
    //This logic needs to be generalized as new multi-output ops are introduced
    if (get_n() == 0)
    {
        adjoints.add_delta(get_inputs().at(0).get_output().get_node(), delta);
    }
}
