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

//================================================================================================
// bfloat16 type
//================================================================================================

#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace ngraph
{
    class bfloat16
    {
    public:
        bfloat16() {}
        bfloat16(float value, bool rounding = false);
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

        static std::vector<float> to_float_vector(const std::vector<bfloat16>&);
        static std::vector<bfloat16> from_float_vector(const std::vector<float>&);

        friend std::ostream& operator<<(std::ostream&, const bfloat16&);

    private:
        uint16_t m_value{0};
    };
}
