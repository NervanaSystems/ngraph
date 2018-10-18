//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include <cmath>
#include <iostream>

#include "ngraph/type/bfloat16.hpp"

using namespace std;
using namespace ngraph;

std::vector<float> bfloat16::to_float_vector(std::vector<bfloat16> v_bf16)
{
    std::vector<float> v_f32(v_bf16.begin(), v_bf16.end());
    return v_f32;
}

std::vector<bfloat16> bfloat16::from_float_vector(std::vector<float> v_f32)
{
    std::vector<bfloat16> v_bf16(v_f32.size());
    for(float a : v_f32)
    {
        v_bf16.push_back(bfloat16(a));
    }
    return v_bf16;
}

std::vector<uint16_t> bfloat16::to_u16_vector(std::vector<bfloat16> v_bf16)
{
    std::vector<uint16_t> v_u16(v_bf16.begin(), v_bf16.end());
    return v_u16;
}

std::vector<bfloat16> bfloat16::from_u16_vector(std::vector<uint16_t> v_u16)
{
    std::vector<bfloat16> v_bf16(v_u16.size());
    for(uint16_t a : v_u16)
    {
        v_bf16.push_back(bfloat16(a));
    }
    return v_bf16;
}

bfloat16::bfloat16(float value)
{
    float* f32_ptr = &value;
    uint16_t* u16_ptr = reinterpret_cast<uint16_t*>(f32_ptr);
    m_value = *u16_ptr;
}

bfloat16::bfloat16(uint16_t value)
{
    m_value = value;
}

bfloat16& bfloat16::operator=(const bfloat16& t)
{
    m_value = t.m_value;
    return *this;
}

const std::string bfloat16::to_string() const
{
    uint32_t u32_value = m_value;
    u32_value = u32_value << 16; 
    float* f32_ptr = reinterpret_cast<float*>(&u32_value);
    float f32_value = *f32_ptr;
    return std::to_string(f32_value);
}

bool bfloat16::operator==(const bfloat16& other) const
{
    return m_value == other.m_value;
}

bool bfloat16::operator<(const bfloat16& other) const
{
    return (int16_t)m_value < (int16_t)(other.m_value);
}

size_t bfloat16::size() const
{
    return sizeof(m_value);
}

bfloat16::operator float() const
{
    uint32_t u32_value = m_value;
    u32_value = u32_value << 16;
    float* f32_ptr = reinterpret_cast<float*>(&u32_value);
    return (*f32_ptr);
}

std::ostream& operator<<(std::ostream& out, const bfloat16& obj)
{
    out << "bfloat16: " << obj.to_string();
    return out;
}
