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

// Contains logic derived from TensorFlow’s bfloat16 implementation
// https://github.com/tensorflow/tensorflow/blob/d354efc/tensorflow/core/lib/bfloat16/bfloat16.h
// Copyright notice from original source file is as follows.

//*******************************************************************************
//  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//    http://www.apache.org/licenses/LICENSE-2.0
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//==============================================================================

#include <cmath>
#include <iostream>

#include "ngraph/type/bfloat16.hpp"

using namespace std;
using namespace ngraph;

// A value represents NaN in bfloat16
static const uint16_t BF16_NAN_VALUE = 0x7FC0;

bool float_isnan(const float& x)
{
    return std::isnan(x);
}

std::vector<float> bfloat16::to_float_vector(const std::vector<bfloat16>& v_bf16)
{
    std::vector<float> v_f32(v_bf16.begin(), v_bf16.end());
    return v_f32;
}

std::vector<bfloat16> bfloat16::from_float_vector(const std::vector<float>& v_f32)
{
    std::vector<bfloat16> v_bf16(v_f32.size());
    for (float a : v_f32)
    {
        v_bf16.push_back(static_cast<bfloat16>(a));
    }
    return v_bf16;
}

bfloat16::bfloat16(float value, RoundingMode mode)
{
    if (float_isnan(value))
    {
        m_value = BF16_NAN_VALUE;
    }
    else if (mode == RoundingMode::TRUNCATE)
    {
        // Truncate off 16 LSB, no rounding
        uint32_t* p = reinterpret_cast<uint32_t*>(&value);
        m_value = static_cast<uint16_t>((*p) >> 16);
    }
    else
    {
        // Rounding with round-nearest-to-even to create bfloat16
        // from float. Refer to TF implementation explanation:
        // https://github.com/tensorflow/tensorflow/blob/d354efc/tensorflow/core/lib/bfloat16/bfloat16.h#L199
        uint32_t* u32_ptr = reinterpret_cast<uint32_t*>(&value);
        uint32_t u32_value = *u32_ptr;
        uint32_t lsb = (u32_value >> 15) & 1;
        uint32_t rounding_bias = 0x7fff + lsb;
        u32_value += rounding_bias;
        m_value = static_cast<uint16_t>(u32_value >> 16);
    }
}

ngraph::bfloat16 bfloat16::from_bits(uint16_t bits)
{
    bfloat16 rc;
    rc.m_value = bits;
    return rc;
}

std::string bfloat16::to_string() const
{
    return std::to_string(static_cast<float>(*this));
}

size_t bfloat16::size() const
{
    return sizeof(m_value);
}

bool bfloat16::operator==(const bfloat16& other) const
{
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfloat-equal"
    return (static_cast<float>(*this) == static_cast<float>(other));
#pragma clang diagnostic pop
}

bool bfloat16::operator<(const bfloat16& other) const
{
    return (static_cast<float>(*this) < static_cast<float>(other));
}

bool bfloat16::operator<=(const bfloat16& other) const
{
    return (static_cast<float>(*this) <= static_cast<float>(other));
}

bool bfloat16::operator>(const bfloat16& other) const
{
    return (static_cast<float>(*this) > static_cast<float>(other));
}

bool bfloat16::operator>=(const bfloat16& other) const
{
    return (static_cast<float>(*this) >= static_cast<float>(other));
}

bfloat16::operator float() const
{
    uint32_t tmp = (static_cast<uint32_t>(m_value) << 16);
    const float* f = reinterpret_cast<const float*>(&tmp);
    return *f;
}

bfloat16::operator double() const
{
    return static_cast<float>(m_value);
}

bfloat16::operator uint16_t() const
{
    return m_value;
}
