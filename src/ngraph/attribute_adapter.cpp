//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include <vector>

#include "ngraph/attribute_adapter.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/strides.hpp"

using namespace std;
using namespace ngraph;

constexpr DiscreteTypeInfo StringAdapter::type_info;

constexpr DiscreteTypeInfo ObjectAdapter<Shape>::type_info;

namespace
{
    template <typename A, typename B>
    A copy_from(B& b)
    {
        A result(b.size());
        for (int i = 0; i < b.size(); ++i)
        {
            result[i] = b[i];
        }
        return result;
    }
}

vector<int64_t> ObjectAdapter<Shape>::get_vector() const
{
    return copy_from<vector<int64_t>>(m_value);
}

void ObjectAdapter<Shape>::set_vector(const vector<int64_t>& value) const
{
    m_value = copy_from<Shape>(value);
}

constexpr DiscreteTypeInfo ObjectAdapter<Strides>::type_info;

vector<int64_t> ObjectAdapter<Strides>::get_vector() const
{
    return copy_from<vector<int64_t>>(m_value);
}

void ObjectAdapter<Strides>::set_vector(const vector<int64_t>& value) const
{
    m_value = copy_from<Strides>(value);
}

constexpr DiscreteTypeInfo ObjectAdapter<AxisSet>::type_info;

vector<int64_t> ObjectAdapter<AxisSet>::get_vector() const
{
    vector<int64_t> result;
    for (auto elt : m_value)
    {
        result.push_back(elt);
    }
    return result;
}

void ObjectAdapter<AxisSet>::set_vector(const vector<int64_t>& value) const
{
    m_value = AxisSet();
    for (auto elt : value)
    {
        m_value.insert(elt);
    }
}
