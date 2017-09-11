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

#include "ngraph/ngraph.hpp"
#include "log.hpp"

using namespace std;
using namespace ngraph;

bool TensorViewType::operator==(const ValueType& that) const
{
    auto that_tvt = dynamic_cast<const TensorViewType*>(&that);
    if (nullptr == that_tvt)
    {
        return false;
    }
    if (that_tvt->get_element_type() != m_element_type)
    {
        return false;
    }
    if (that_tvt->get_shape() != m_shape)
    {
        return false;
    }
    return true;
}

void TensorViewType::collect_tensor_views(std::vector<std::shared_ptr<const TensorViewType>>& views) const
{
    views.push_back(shared_from_this());
}

bool TupleType::operator==(const ValueType& that) const
{
    auto that_tvt = dynamic_cast<const TupleType*>(&that);
    if (nullptr == that_tvt)
    {
        return false;
    }
    return that_tvt->get_element_types() == get_element_types();
}

void TupleType::collect_tensor_views(std::vector<std::shared_ptr<const TensorViewType>>& views) const
{
    for(auto elt : m_element_types)
    {
        elt->collect_tensor_views(views);
    }
}

std::ostream& ngraph::operator<<(std::ostream& out, const ValueType& obj)
{
    out << "ValueType()";
    return out;
}

std::ostream& ngraph::operator<<(std::ostream& out, const TensorViewType& obj)
{
    out << "TensorViewType(" << obj.m_element_type << ", " << obj.m_shape  << ")";
    return out;
}

std::ostream& ngraph::operator<<(std::ostream& out, const TupleType& obj)
{
    out << "TupleType()";
    return out;
}
