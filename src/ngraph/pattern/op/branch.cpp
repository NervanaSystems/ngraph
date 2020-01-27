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

#include "ngraph/pattern/op/branch.hpp"
#include "ngraph/pattern/matcher.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo pattern::op::Branch::type_info;

const NodeTypeInfo& pattern::op::Branch::get_type_info() const
{
    return type_info;
}

bool pattern::op::Branch::match_value(Matcher* matcher,
                                      const Output<Node>& pattern_value,
                                      const Output<Node>& graph_value)
{
    return matcher->match_value(get_destination(), graph_value);
}
