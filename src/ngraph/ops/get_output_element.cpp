// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <sstream>

#include "ngraph/ops/get_output_element.hpp"

using namespace std;
using namespace ngraph;

op::GetOutputElement::GetOutputElement(const std::shared_ptr<Node>& arg, size_t n)
    : Node("GetOutputElement", {arg})
    , m_n{n}
{
    if (arg->get_outputs().size() < 1)
    {
        throw ngraph_error("Argument should have at least one output tensor");
    }

    if (m_n >= arg->get_outputs().size())
    {
        throw ngraph_error("Indexing tuple beyond its size");
    }

    auto& output = arg->get_outputs().at(n);
    set_value_type_checked(output.get_element_type(), output.get_shape());
}
