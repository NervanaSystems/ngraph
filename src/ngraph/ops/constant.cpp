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

#include <cstdio>

#include "ngraph/log.hpp"
#include "ngraph/ops/constant.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

op::Constant::~Constant()
{
    if (m_data)
    {
        ngraph::aligned_free(m_data);
    }
}

std::vector<std::string> op::Constant::get_value_strings() const
{
    vector<string> rc;

    if (m_element_type == element::boolean)
    {
        for (int value : get_vector<char>())
        {
            rc.push_back(to_string(value));
        }
    }
    else if (m_element_type == element::f32)
    {
        for (float value : get_vector<float>())
        {
            rc.push_back(to_string(value));
        }
    }
    else if (m_element_type == element::f64)
    {
        for (double value : get_vector<double>())
        {
            rc.push_back(to_string(value));
        }
    }
    else if (m_element_type == element::i8)
    {
        for (int value : get_vector<int8_t>())
        {
            rc.push_back(to_string(value));
        }
    }
    else if (m_element_type == element::i16)
    {
        for (int value : get_vector<int16_t>())
        {
            rc.push_back(to_string(value));
        }
    }
    else if (m_element_type == element::i32)
    {
        for (int32_t value : get_vector<int32_t>())
        {
            NGRAPH_INFO << value;
            rc.push_back(to_string(value));
        }
    }
    else if (m_element_type == element::i64)
    {
        for (int64_t value : get_vector<int64_t>())
        {
            rc.push_back(to_string(value));
        }
    }
    else if (m_element_type == element::u8)
    {
        for (uint value : get_vector<uint8_t>())
        {
            rc.push_back(to_string(value));
        }
    }
    else if (m_element_type == element::u16)
    {
        for (uint value : get_vector<uint16_t>())
        {
            rc.push_back(to_string(value));
        }
    }
    else if (m_element_type == element::u32)
    {
        for (uint32_t value : get_vector<uint32_t>())
        {
            rc.push_back(to_string(value));
        }
    }
    else if (m_element_type == element::u64)
    {
        for (uint64_t value : get_vector<uint64_t>())
        {
            rc.push_back(to_string(value));
        }
    }
    else
    {
        throw std::runtime_error("unsupported type");
    }

    return rc;
}

template <>
void op::Constant::write_to_buffer<std::string>(const element::Type& target_type,
                                                const Shape& target_shape,
                                                const std::vector<std::string>& source,
                                                void* target,
                                                size_t target_element_count)
{
}
