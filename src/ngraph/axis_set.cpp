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

#include "ngraph/axis_set.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

std::ostream& ngraph::operator<<(std::ostream& s, const AxisSet& axis_set)
{
    s << "AxisSet{";
    s << ngraph::join(axis_set);
    s << "}";
    return s;
}

NGRAPH_API constexpr DiscreteTypeInfo AttributeAdapter<AxisSet>::type_info;

const vector<int64_t>& AttributeAdapter<AxisSet>::get()
{
    if (!m_buffer_valid)
    {
        for (auto elt : m_value)
        {
            m_buffer.push_back(elt);
        }
    }
    return m_buffer;
}

void AttributeAdapter<AxisSet>::set(const vector<int64_t>& value)
{
    m_value = AxisSet();
    for (auto elt : value)
    {
        m_value.insert(elt);
    }
    m_buffer_valid = false;
}
