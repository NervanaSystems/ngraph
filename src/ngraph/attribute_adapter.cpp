//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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
#include "ngraph/coordinate.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/partial_shape.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/strides.hpp"
#include "ngraph/type.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    constexpr DiscreteTypeInfo AttributeAdapter<float>::type_info;
    const double& AttributeAdapter<float>::get()
    {
        if (!m_buffer_valid)
        {
            m_buffer = m_value;
            m_buffer_valid = true;
        }
        return m_buffer;
    }

    void AttributeAdapter<float>::set(const double& value)
    {
        m_value = value;
        m_buffer_valid = false;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<double>::type_info;
    const double& AttributeAdapter<double>::get()
    {
        if (!m_buffer_valid)
        {
            m_buffer = m_value;
            m_buffer_valid = true;
        }
        return m_buffer;
    }

    void AttributeAdapter<double>::set(const double& value)
    {
        m_value = value;
        m_buffer_valid = false;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<int8_t>::type_info;
    const int64_t& AttributeAdapter<int8_t>::get()
    {
        if (!m_buffer_valid)
        {
            m_buffer = m_value;
            m_buffer_valid = true;
        }
        return m_buffer;
    }

    void AttributeAdapter<int8_t>::set(const int64_t& value)
    {
        m_value = value;
        m_buffer_valid = false;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<int16_t>::type_info;
    const int64_t& AttributeAdapter<int16_t>::get()
    {
        if (!m_buffer_valid)
        {
            m_buffer = m_value;
            m_buffer_valid = true;
        }
        return m_buffer;
    }

    void AttributeAdapter<int16_t>::set(const int64_t& value)
    {
        m_value = value;
        m_buffer_valid = false;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<int32_t>::type_info;
    const int64_t& AttributeAdapter<int32_t>::get()
    {
        if (!m_buffer_valid)
        {
            m_buffer = m_value;
            m_buffer_valid = true;
        }
        return m_buffer;
    }

    void AttributeAdapter<int32_t>::set(const int64_t& value)
    {
        m_value = value;
        m_buffer_valid = false;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<int64_t>::type_info;
    const int64_t& AttributeAdapter<int64_t>::get()
    {
        if (!m_buffer_valid)
        {
            m_buffer = m_value;
            m_buffer_valid = true;
        }
        return m_buffer;
    }

    void AttributeAdapter<int64_t>::set(const int64_t& value)
    {
        m_value = value;
        m_buffer_valid = false;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<uint8_t>::type_info;
    const int64_t& AttributeAdapter<uint8_t>::get()
    {
        if (!m_buffer_valid)
        {
            m_buffer = m_value;
            m_buffer_valid = true;
        }
        return m_buffer;
    }

    void AttributeAdapter<uint8_t>::set(const int64_t& value)
    {
        m_value = value;
        m_buffer_valid = false;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<uint16_t>::type_info;
    const int64_t& AttributeAdapter<uint16_t>::get()
    {
        if (!m_buffer_valid)
        {
            m_buffer = m_value;
            m_buffer_valid = true;
        }
        return m_buffer;
    }

    void AttributeAdapter<uint16_t>::set(const int64_t& value)
    {
        m_value = value;
        m_buffer_valid = false;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<uint32_t>::type_info;
    const int64_t& AttributeAdapter<uint32_t>::get()
    {
        if (!m_buffer_valid)
        {
            m_buffer = m_value;
            m_buffer_valid = true;
        }
        return m_buffer;
    }

    void AttributeAdapter<uint32_t>::set(const int64_t& value)
    {
        m_value = value;
        m_buffer_valid = false;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<uint64_t>::type_info;
    const int64_t& AttributeAdapter<uint64_t>::get()
    {
        if (!m_buffer_valid)
        {
            m_buffer = m_value;
            m_buffer_valid = true;
        }
        return m_buffer;
    }

    void AttributeAdapter<uint64_t>::set(const int64_t& value)
    {
        m_value = value;
        m_buffer_valid = false;
    }

#ifdef __APPLE__
    // size_t is not uint_64t on OSX
    constexpr DiscreteTypeInfo AttributeAdapter<size_t>::type_info;
    const int64_t& AttributeAdapter<size_t>::get()
    {
        if (!m_buffer_valid)
        {
            m_buffer = m_value;
            m_buffer_valid = true;
        }
        return m_buffer;
    }

    void AttributeAdapter<size_t>::set(const int64_t& value)
    {
        m_value = value;
        m_buffer_valid = false;
    }
#endif

    constexpr DiscreteTypeInfo AttributeAdapter<vector<int64_t>>::type_info;

    const vector<int64_t>& AttributeAdapter<vector<int64_t>>::get() { return m_value; }
    void AttributeAdapter<vector<int64_t>>::set(const vector<int64_t>& value) { m_value = value; }
    constexpr DiscreteTypeInfo AttributeAdapter<vector<uint64_t>>::type_info;

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
}
