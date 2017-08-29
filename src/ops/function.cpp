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

#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;

Parameter::Parameter(Function& function, size_t index)
    : Node({})
    , m_function(function)
    , m_index(index)
{
}

void Parameter::propagate_types()
{
    if (m_type == nullptr)
    {
        throw ngraph_error{"Unitialized parameter"};
    }
}

Function::Function(size_t n_parameters)
    : m_parameters(n_parameters)
    , m_name("Function")
{
    for (int i = 0; i < n_parameters; i++)
    {
        m_parameters[i] = std::make_shared<Parameter>(*this, i);
    }
}
