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

#include "ngraph/except.hpp"
#include "ngraph/log.hpp"
#include "ngraph/types/type.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

bool ValueType::operator!=(const ValueType& that) const
{
    return !(*this == that);
}

bool TensorViewType::operator==(const ValueType& that) const
{
    bool rc = true;
    auto that_tvt = dynamic_cast<const TensorViewType*>(&that);
    auto that_tt = dynamic_cast<const TupleType*>(&that);
    if (that_tvt != nullptr)
    {
        rc = true;
        if (that_tvt->get_element_type() != m_element_type)
        {
            rc = false;
        }
        if (that_tvt->get_shape() != m_shape)
        {
            rc = false;
        }
    }
    else if (that_tt != nullptr)
    {
        rc = *that_tt == *this;
    }
    return rc;
}

void TensorViewType::collect_tensor_views(
    std::vector<std::shared_ptr<const TensorViewType>>& views) const
{
    views.push_back(shared_from_this());
}

bool TupleType::operator==(const ValueType& that) const
{
    auto that_tvt = dynamic_cast<const TupleType*>(&that);
    if (that_tvt == nullptr)
    {
        return false;
    }

    vector<shared_ptr<const ValueType>> this_values = this->get_element_types();
    vector<shared_ptr<const ValueType>> that_values = that_tvt->get_element_types();
    bool rc = this_values.size() == that_values.size();
    if (rc)
    {
        for (size_t i = 0; i < this_values.size(); i++)
        {
            rc &= this_values[i]->get_element_type() == that_values[i]->get_element_type();
        }
    }

    return rc;
}

void TupleType::collect_tensor_views(
    std::vector<std::shared_ptr<const TensorViewType>>& views) const
{
    for (auto elt : m_element_types)
    {
        elt->collect_tensor_views(views);
    }
}

const Shape& TupleType::get_shape() const
{
    throw ngraph_error("get_shape() called on Tuple");
}

const element::Type& TupleType::get_element_type() const
{
    throw ngraph_error("get_element_type() called on Tuple");
}

std::ostream& ngraph::operator<<(std::ostream& out, const ValueType& obj)
{
    auto tvt = dynamic_cast<const TensorViewType*>(&obj);
    auto tup = dynamic_cast<const TupleType*>(&obj);

    if (tvt != nullptr)
    {
        out << *tvt;
    }
    else if (tup != nullptr)
    {
        out << *tup;
    }
    else
    {
        out << "ValueType()";
    }
    return out;
}

std::ostream& ngraph::operator<<(std::ostream& out, const TensorViewType& obj)
{
    out << "TensorViewType(" << obj.m_element_type << ", {" << join(obj.m_shape) << "})";
    return out;
}

std::ostream& ngraph::operator<<(std::ostream& out, const TupleType& obj)
{
    out << "TupleType()";
    return out;
}
