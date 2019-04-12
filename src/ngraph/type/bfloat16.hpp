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

#pragma once

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#define ROUND_MODE_TO_NEAREST_EVEN

namespace ngraph
{
    class bfloat16
    {
    public:
        bfloat16()
            : m_value{0}
        {
        }
        bfloat16(float value)
            : m_value
        {
#if defined ROUND_MODE_TO_NEAREST
            round_to_nearest(value)
#elif defined ROUND_MODE_TO_NEAREST_EVEN
            round_to_nearest_even(value)
#elif defined ROUND_MODE_TRUNCATE
            truncate(value)
#else
#error                                                                                             \
    "ROUNDING_MODE must be one of ROUND_MODE_TO_NEAREST, ROUND_MODE_TO_NEAREST_EVEN, or ROUND_MODE_TRUNCATE"
#endif
        }
        {
        }

        std::string to_string() const;
        size_t size() const;
        bool operator==(const bfloat16& other) const;
        bool operator!=(const bfloat16& other) const { return !(*this == other); }
        bool operator<(const bfloat16& other) const;
        bool operator<=(const bfloat16& other) const;
        bool operator>(const bfloat16& other) const;
        bool operator>=(const bfloat16& other) const;
        operator float() const;
        operator double() const;

        static std::vector<float> to_float_vector(const std::vector<bfloat16>&);
        static std::vector<bfloat16> from_float_vector(const std::vector<float>&);
        static bfloat16 from_bits(uint16_t bits) { return bfloat16(bits, false); }
        uint16_t to_bits() const;
        friend std::ostream& operator<<(std::ostream& out, const bfloat16& obj)
        {
            out << static_cast<float>(obj);
            return out;
        }

#define cu32(x) (F32(x).i)

        static uint16_t round_to_nearest_even(float x)
        {
            return static_cast<uint16_t>((cu32(x) + ((cu32(x) & 0x00010000) >> 1)) >> 16);
        }

        static uint16_t round_to_nearest(float x)
        {
            return static_cast<uint16_t>((cu32(x) + 0x8000) >> 16);
        }

        static uint16_t truncate(float x) { return static_cast<uint16_t>((cu32(x)) >> 16); }
    private:
        union F32 {
            F32(float val)
                : f{val}
            {
            }
            F32(uint32_t val)
                : i{val}
            {
            }
            float f;
            uint32_t i;
        };
        // This should be private since it is ugly. Need the bool so the signature can't match
        // the float version of the ctor.
        bfloat16(uint16_t value, bool)
            : m_value{value}
        {
        }

        uint16_t m_value;

        static uint16_t BF16_NAN_VALUE;
    };
}
