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

#include <ngraph/ngraph.hpp>

using namespace std;
using namespace ngraph;

bool TensorViewType::operator==(const ValueType::ptr& that) const
{
    auto that_tvt = dynamic_pointer_cast<TensorViewType>(that);
    if (nullptr == that_tvt)
    {
        return false;
    }
    if (that_tvt->element_type() != m_element_type)
    {
        return false;
    }
    if (that_tvt->shape() != m_shape)
    {
        return false;
    }
    return true;
}

bool TupleType::operator==(const ValueType::ptr& that) const
{
    auto that_tvt = dynamic_pointer_cast<TupleType>(that);
    if (nullptr == that_tvt)
    {
        return false;
    }
    return that_tvt->element_types() == element_types();
}
