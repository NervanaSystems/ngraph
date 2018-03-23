/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <cmath>
#include <cstdio>

#include "ngraph/log.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

template <typename T>
std::string to_cpp_string(T value)
{
    string rc;
    if (std::isnan(value))
    {
        rc = "NAN";
    }
    else if (std::isinf(value))
    {
        if (value > 0)
        {
            rc = "INFINITY";
        }
        else
        {
            rc = "-INFINITY";
        }
    }
    else
    {
        stringstream ss;
        ss << value;
        rc = ss.str();
    }
    return rc;
}

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
            rc.push_back(to_cpp_string(value));
        }
    }
    else if (m_element_type == element::f64)
    {
        for (double value : get_vector<double>())
        {
            rc.push_back(to_cpp_string(value));
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

//
// We have to open up namespace blocks here to work around a problem with gcc:
//
// https://stackoverflow.com/questions/25594644/warning-specialization-of-template-in-different-namespace
//
namespace ngraph
{
    namespace op
    {
        template <>
        void Constant::write_to_buffer<std::string>(const element::Type& target_type,
                                                    const Shape& target_shape,
                                                    const std::vector<std::string>& source,
                                                    void* target,
                                                    size_t target_element_count)
        {
        }
    }
}
