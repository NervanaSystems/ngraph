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

Parameter::Parameter(const ValueType::ptr& value_type)
    : Node({}, value_type)
    , m_function(nullptr)
    , m_index(0)
{
}

void Parameter::assign_function(Function* function, size_t index)
{
    if (nullptr != m_function)
    {
        throw ngraph_error("Re-assigning function to a parameter.");
    }
    m_function = function;
    m_index    = index;
}

void Parameter::propagate_types() {}

shared_ptr<Parameter> ngraph::op::parameter(const ValueType::ptr& value_type)
{
    return make_shared<Parameter>(value_type);
}

shared_ptr<Parameter> ngraph::op::parameter(const ngraph::element::Type element_type,
                                            const Shape&                shape)
{
    return make_shared<Parameter>(make_shared<TensorViewType>(element_type, shape));
}
