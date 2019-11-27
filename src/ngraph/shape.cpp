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

#include "ngraph/shape.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

std::ostream& ngraph::operator<<(std::ostream& s, const Shape& shape)
{
    s << "Shape{";
    s << ngraph::join(shape);
    s << "}";
    return s;
}

NGRAPH_API constexpr DiscreteTypeInfo AttributeAdapter<Shape>::type_info;

const vector<int64_t>& AttributeAdapter<Shape>::get()
{
    if (!m_buffer_valid)
    {
        m_buffer = copy_from<vector<int64_t>>(m_value);
        m_buffer_valid = true;
    }
    return m_buffer;
}

void AttributeAdapter<Shape>::set(const vector<int64_t>& value)
{
    m_value = copy_from<Shape>(value);
    m_buffer_valid = false;
}
