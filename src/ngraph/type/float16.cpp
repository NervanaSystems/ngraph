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

static bool float_isnan(const float& x)
{
    return std::isnan(x);
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
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wfloat-equal"
    return (static_cast<float>(*this) == static_cast<float>(other));
#pragma clang diagnostic pop
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
    unsigned exp = 0x1F & (m_value >> 10);
    unsigned fexp = exp + 127 - 15;
    unsigned frac = m_value & 0x03FF;
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
    else if (frac == 0x1F)
    {
        fexp = 0xFF;
    }
    frac = frac << (24 - 8);
    i_val = (m_value & 0x8000) << 31 | (fexp << 24) | frac;
    return f_val;
}

uint16_t float16::to_bits() const
{
    return m_value;
}
