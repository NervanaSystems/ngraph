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

#include "ngraph/module.hpp"
#include "ngraph/function.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

Module::Module()
{
}

Module::~Module()
{
}

void Module::add_function(shared_ptr<Function> func)
{
    m_functions.push_back(func);

    traverse_nodes(func->get_result(), [](Node* node)
    {
    });
}

vector<shared_ptr<Function>>& Module::get_functions()
{
    return m_functions;
}

const vector<shared_ptr<Function>>& Module::get_functions() const
{
    return m_functions;
}

