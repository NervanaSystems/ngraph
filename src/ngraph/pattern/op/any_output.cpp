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

#include "ngraph/pattern/op/any_output.hpp"
#include "ngraph/log.hpp"
#include "ngraph/pattern/matcher.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo pattern::op::AnyOutput::type_info;

const NodeTypeInfo& pattern::op::AnyOutput::get_type_info() const
{
    return type_info;
}

pattern::op::AnyOutput::AnyOutput(const std::shared_ptr<Node>& pattern)
    : Pattern(pattern->outputs())
{
    NGRAPH_INFO << pattern->description() << " " << pattern->outputs().size();
}

bool pattern::op::AnyOutput::match_value(Matcher* matcher,
                                         const Output<Node>& pattern_value,
                                         const Output<Node>& graph_value)
{
    NGRAPH_INFO << "******************** " << pattern_value;
    NGRAPH_INFO << "******************** " << graph_value;
    NGRAPH_INFO << get_input_size();
    bool rc = false;
    for (Output<Node> output : input_values())
    {
        // NGRAPH_INFO << output.get_node();
        NGRAPH_INFO;
        if (output.get_node()->match_value(matcher, output, graph_value))
        {
            NGRAPH_INFO << "*********";
            rc = true;
            break;
        }
    }
    return rc;
    // return input_value(0).get_node()->match_node(matcher, graph_value);
}
