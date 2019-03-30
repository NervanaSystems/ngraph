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
        bfloat16() {}
        bfloat16(float value, RoundingMode mode = RoundingMode::ROUND);

        bfloat16(const bfloat16&) = default;
        bfloat16& operator=(const bfloat16&) = default;
        virtual ~bfloat16() {}
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
            constexpr F32()
                : i{0}
            {
            }
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

        uint16_t m_value{0};

        static constexpr uint16_t BF16_NAN_VALUE = 0x7FC0;
    };
}
