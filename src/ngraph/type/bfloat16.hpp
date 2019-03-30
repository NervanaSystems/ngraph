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

//================================================================================================
// bfloat16 type
//================================================================================================

#pragma once

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace ngraph
{
    class bfloat16
    {
    public:
        enum class RoundingMode
        {
            TRUNCATE,
            ROUND
        };
        constexpr bfloat16()
            : m_value{0}
        {
        }
        constexpr bfloat16(float value, RoundingMode mode = RoundingMode::ROUND)
            : m_value{(
                  std::isnan(value)
                      ? BF16_NAN_VALUE
                      : (mode == RoundingMode::TRUNCATE
                             ?
                             // Truncate off 16 LSB, no rounding
                             static_cast<uint16_t>((F32(value).i) >> 16)
                             :
                             // Rounding with round-nearest-to-even to create bfloat16
                             // from float. Refer to TF implementation explanation:
                             // https://github.com/tensorflow/tensorflow/blob/d354efc/tensorflow/core/lib/bfloat16/bfloat16.h#L199
                             static_cast<uint16_t>(
                                 (F32(value).i + (0x7fff + ((F32(value).i >> 15) & 1))) >> 16)))}
        {
            // The initialization of m_value above is ugly so I have included the original source
            // that was used to create that monstrosity.
            // This was done to add a c++11 constexpr ctor
            // if (std::isnan(value))
            // {
            //     m_value = BF16_NAN_VALUE;
            // }
            // else if (mode == RoundingMode::TRUNCATE)
            // {
            //     // Truncate off 16 LSB, no rounding
            //     uint32_t* p = reinterpret_cast<uint32_t*>(&value);
            //     m_value = static_cast<uint16_t>((*p) >> 16);
            // }
            // else
            // {
            //     // Rounding with round-nearest-to-even to create bfloat16
            //     // from float. Refer to TF implementation explanation:
            //     // https://github.com/tensorflow/tensorflow/blob/d354efc/tensorflow/core/lib/bfloat16/bfloat16.h#L199
            //     uint32_t* u32_ptr = reinterpret_cast<uint32_t*>(&value);
            //     uint32_t u32_value = *u32_ptr;
            //     uint32_t lsb = (u32_value >> 15) & 1;
            //     uint32_t rounding_bias = 0x7fff + lsb;
            //     u32_value += rounding_bias;
            //     m_value = static_cast<uint16_t>(u32_value >> 16);
            // }
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
        operator uint16_t() const;

        static std::vector<float> to_float_vector(const std::vector<bfloat16>&);
        static std::vector<bfloat16> from_float_vector(const std::vector<float>&);
        static bfloat16 from_bits(uint16_t bits);

        friend std::ostream& operator<<(std::ostream& out, const bfloat16& obj)
        {
            out << static_cast<float>(obj);
            return out;
        }

    private:
        union F32 {
            constexpr F32(float val)
                : f{val}
            {
            }
            constexpr F32(uint32_t val)
                : i{val}
            {
            }
            float f;
            uint32_t i;
        };

        uint16_t m_value;

        static constexpr uint16_t BF16_NAN_VALUE = 0x7FC0;
    };
}
