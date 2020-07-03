//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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
#include "ngraph/util.hpp"
#include "ngraph/log.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::GetOutputElement::type_info;

op::GetOutputElement::GetOutputElement(const shared_ptr<Node>& arg, size_t n)
    : Op({arg->output(n)})
    , m_n{n}
{
    NGRAPH_INFO << "GetOutputElement ctor *******************************************";
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

    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

shared_ptr<Node> op::GetOutputElement::clone_with_new_inputs(const OutputVector& inputs) const
{
    auto& value = inputs.at(0);
    return make_shared<op::GetOutputElement>(value.get_node_shared_ptr(), value.get_index());
}

Output<Node> op::GetOutputElement::get_as_output() const
{
    return input_value(0);
}

NodeVector op::GetOutputElement::get_arguments() const
{
    return NodeVector{input_value(0).get_node_shared_ptr()};
}

void op::GetOutputElement::generate_adjoints(autodiff::Adjoints& adjoints,
                                             const OutputVector& deltas)
{
    auto delta = deltas.at(0);

    adjoints.add_delta(input_value(0), delta);
}

std::ostream& op::GetOutputElement::write_description(std::ostream& out, uint32_t depth) const
{
    if (depth == 0)
    {
        out << get_name() << "(" << input_value(0).get_node()->get_name() << "["
            << input_value(0).get_index() << "])";
    }
    else
    {
        out << "v" << get_type_info().version << "::" << get_type_info().name << " " << get_name()
            << "(" << join(input_values()) << ") -> (";
        out << get_output_element_type(0) << get_output_partial_shape(0);
        out << ")";
    }
    return out;
}

bool op::GetOutputElement::match_value(pattern::Matcher* matcher,
                                       const Output<Node>& pattern_value,
                                       const Output<Node>& graph_value)
{
    NGRAPH_INFO;
    return Node::match_value(matcher, pattern_value, graph_value);
}

bool op::GetOutputElement::match_node(pattern::Matcher* matcher, const Output<Node>& graph_value)
{
    NGRAPH_INFO;
    return Node::match_node(matcher, graph_value);
}
