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

#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/matcher.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo pattern::op::Label::type_info;

const NodeTypeInfo& pattern::op::Label::get_type_info() const
{
    return type_info;
}

bool pattern::op::Label::match_value(Matcher* matcher,
                                     const Output<Node>& pattern_value,
                                     const Output<Node>& graph_value)
{
    auto& pattern_map = matcher->get_pattern_value_map();
    if (pattern_map.count(shared_from_this()))
    {
        if (pattern_map[shared_from_this()] == graph_value)
        {
            matcher->add_node(graph_value);
            return true;
        };
        return false;
    }
    if (m_predicate(graph_value))
    {
        if (0 == get_input_size())
        {
            matcher->add_node(graph_value);
            pattern_map[shared_from_this()] = graph_value;
            return true;
        }
        for (auto input_value : input_values())
        {
            auto saved = matcher->start_match();
            matcher->add_node(graph_value);
            if (matcher->match_value(input_value, graph_value))
            {
                pattern_map[shared_from_this()] = graph_value;
                return saved.finish(true);
            }
        }
    }
    return false;
}
