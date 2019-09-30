//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include <sstream>

#include "ngraph/op/get_output_element.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::GetOutputElement::type_info;

op::GetOutputElement::GetOutputElement(const shared_ptr<Node>& arg, size_t n)
    : Op({Output<Node>{arg, n}})
    , m_n{n}
{
    constructor_validate_and_infer_types();
}

void op::GetOutputElement::validate_and_infer_types()
{
    NODE_VALIDATION_CHECK(this,
                          m_n < input_value(0).get_node()->get_output_size(),
                          "Output at index ",
                          m_n,
                          " requested, but node has only ",
                          get_input_size(),
                          " inputs.");

    set_output_type(0, input(0).get_element_type(), input(0).get_partial_shape());
}

shared_ptr<Node> op::GetOutputElement::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<GetOutputElement>(new_args.at(0), m_n);
}

Output<Node> op::GetOutputElement::get_as_output() const
{
    return input_value(0);
}

NodeVector op::GetOutputElement::get_arguments() const
{
    return NodeVector{input_value(0).get_node_shared_ptr()};
}

void op::GetOutputElement::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    adjoints.add_delta(input_value(0).get_node_shared_ptr(), delta, get_n());
}

NodeVector op::get_output_elements(const shared_ptr<Node>& mon)
{
    NodeVector goes(mon->get_output_size());
    for (auto o : mon->outputs())
    {
        goes.at(o.get_index()) = o.as_single_output_node();
    }
    return goes;
}
