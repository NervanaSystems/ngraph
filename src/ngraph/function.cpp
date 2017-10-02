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

#include <memory>

#include "ngraph/function.hpp"

using namespace std;
using namespace ngraph;

Function::Function(const std::shared_ptr<Node>&                       result,
                   const std::vector<std::shared_ptr<op::Parameter>>& parameters,
                   const std::shared_ptr<ValueType>&                  result_type)
    : m_result(result)
    , m_parameters(parameters)
    , m_name("Function")
    , m_result_type(result_type)
{
    size_t i = 0;
    for (auto parameter : parameters)
    {
        parameter->assign_function(this, i++);
    }
}
