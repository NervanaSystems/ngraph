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
// https://github.com/tensorflow/tensorflow/blob/d354efc/tensorflow/core/lib/float16/float16.h
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
#include <limits>

#include "ngraph/type/float16.hpp"

using namespace std;
using namespace ngraph;

static_assert(sizeof(float16) == 2, "class float16 must be exactly 2 bytes");

float16::float16(float value)
{
    union {
        float fv;
        uint32_t iv;
    };
    fv = value;
    uint32_t sign = iv & 0x80000000;
    uint32_t biased_exp = (iv & 0x7F800000) >> 23;
    uint32_t raw_frac = (iv & 0x007FFFFF);
    int32_t exp = biased_exp - 127;
    int32_t min_exp = -14 - frac_size;
    if (biased_exp == 0 || exp < min_exp)
    {
        // Goes to 0
        biased_exp = 0;
    }
    else if (biased_exp == 0xFF)
    {
        // Infinity or NAN.
        biased_exp = 0x1F;
        raw_frac = raw_frac >> (23 - frac_size);
    }
    else if (exp < -14)
    {
        // denorm or 0
        biased_exp = 0;
        raw_frac |= 0x00800000;
        raw_frac = raw_frac >> (exp + 16);
    }
    else if (exp > 15)
    {
        biased_exp = 0x1F;
        raw_frac = 0;
    }
    else
    {
        raw_frac = raw_frac >> (23 - frac_size);
        biased_exp = exp + exp_bias;
    }
    m_value = (sign >> 16) | (biased_exp << frac_size) | raw_frac;
}

std::string float16::to_string() const
{
    return std::to_string(static_cast<float>(*this));
}

size_t float16::size() const
{
    return sizeof(m_value);
}

bool float16::operator==(const float16& other) const
{
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
    return (static_cast<float>(*this) == static_cast<float>(other));
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
}

bool float16::operator<(const float16& other) const
{
    return (static_cast<float>(*this) < static_cast<float>(other));
}

bool float16::operator<=(const float16& other) const
{
    return (static_cast<float>(*this) <= static_cast<float>(other));
}

bool float16::operator>(const float16& other) const
{
    return (static_cast<float>(*this) > static_cast<float>(other));
}

bool float16::operator>=(const float16& other) const
{
    return (static_cast<float>(*this) >= static_cast<float>(other));
}

float16::operator float() const
{
    union {
        uint32_t i_val;
        float f_val;
    };
    uint32_t exp = 0x1F & (m_value >> frac_size);
    uint32_t fexp = exp + 127 - 15;
    uint32_t frac = m_value & 0x03FF;
    if (exp == 0)
    {
        if (frac == 0)
        {
            fexp = 0;
        }
        else
        {
            // Normalize
            fexp++;
            while (0 == (frac & 0x0400))
            {
                fexp--;
                frac = frac << 1;
            }
            frac &= 0x03FF;
        }
    }
    else if (exp == 0x1F)
    {
        fexp = 0xFF;
    }
    frac = frac << (23 - frac_size);
    i_val = static_cast<uint32_t>((m_value & 0x8000)) << 16 | (fexp << 23) | frac;
    return f_val;
}

uint16_t float16::to_bits() const
{
    return m_value;
}
