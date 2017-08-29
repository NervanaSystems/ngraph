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

Function::Function(const Node::ptr&                                       result,
                   const std::vector<std::shared_ptr<ngraph::Parameter>>& parameters)
    : m_result(result)
    , m_parameters(parameters)
    , m_name("Function")
{
    size_t i = 0;
    for (auto parameter : parameters)
    {
        parameter->assign_function(this, i++);
    }
}

shared_ptr<Function> ngraph::op::function(const Node::ptr&                               result,
                                          const initializer_list<shared_ptr<Parameter>>& parameters)
{
    return make_shared<Function>(result, parameters);
}

shared_ptr<Function> ngraph::op::function(const Node::ptr&                     result,
                                          const vector<shared_ptr<Parameter>>& parameters)
{
    return make_shared<Function>(result, parameters);
}
