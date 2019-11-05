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
#include "ngraph/partial_shape.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/strides.hpp"
#include "ngraph/type.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

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

namespace ngraph
{
    NGRAPH_API constexpr DiscreteTypeInfo AttributeAdapter<vector<int64_t>>::type_info;

    const vector<int64_t>& AttributeAdapter<vector<int64_t>>::get() { return m_value; }
    void AttributeAdapter<vector<int64_t>>::set(const vector<int64_t>& value) { m_value = value; }
    NGRAPH_API constexpr DiscreteTypeInfo AttributeAdapter<vector<uint64_t>>::type_info;

    const vector<int64_t>& AttributeAdapter<vector<uint64_t>>::get()
    {
        if (!m_buffer_valid)
        {
            m_buffer = copy_from<vector<int64_t>>(m_value);
            m_buffer_valid = true;
        }
        return m_buffer;
    }

    void AttributeAdapter<vector<uint64_t>>::set(const vector<int64_t>& value)
    {
        m_value = copy_from<vector<uint64_t>>(value);
        m_buffer_valid = false;
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

    NGRAPH_API constexpr DiscreteTypeInfo AttributeAdapter<Strides>::type_info;

    const vector<int64_t>& AttributeAdapter<Strides>::get()
    {
        if (!m_buffer_valid)
        {
            m_buffer = copy_from<vector<int64_t>>(m_value);
            m_buffer_valid = true;
        }
        return m_buffer;
    }

    void AttributeAdapter<Strides>::set(const vector<int64_t>& value)
    {
        m_value = copy_from<Strides>(value);
        m_buffer_valid = false;
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

    NGRAPH_API constexpr DiscreteTypeInfo AttributeAdapter<PartialShape>::type_info;
    NGRAPH_API constexpr DiscreteTypeInfo AttributeAdapter<element::Type>::type_info;
    NGRAPH_API constexpr DiscreteTypeInfo AttributeAdapter<op::AutoBroadcastSpec>::type_info;
}
