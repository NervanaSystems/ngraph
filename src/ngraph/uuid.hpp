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

#pragma once

#include <array>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

static std::mt19937_64 random_generator;

namespace ngraph
{
    class uuid_type;
}

class ngraph::uuid_type
{
public:
    uuid_type()
    {
        m_data[0] = random_generator();
        m_data[1] = random_generator();
        uint8_t* p = reinterpret_cast<uint8_t*>(m_data);
        p[6] = (p[6] & 0x0F) | 0x40;
        p[8] = (p[8] & 0x3F) | 0x80;
    }

    std::string to_string() const
    {
        std::stringstream ss;
        const uint8_t* p = reinterpret_cast<const uint8_t*>(m_data);
        for (int i = 0; i < 4; i++)
        {
            ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(*p++);
        }
        ss << "-";
        for (int i = 0; i < 2; i++)
        {
            ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(*p++);
        }
        ss << "-";
        for (int i = 0; i < 2; i++)
        {
            ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(*p++);
        }
        ss << "-";
        for (int i = 0; i < 2; i++)
        {
            ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(*p++);
        }
        ss << "-";
        for (int i = 0; i < 6; i++)
        {
            ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(*p++);
        }
        return ss.str();
    }

    static uuid_type zero()
    {
        uuid_type rc;
        rc.m_data[0] = 0;
        rc.m_data[1] = 0;
        return rc;
    }

    bool operator==(const uuid_type& other) const { return memcmp(m_data, other.m_data, 16) == 0; }
    bool operator!=(const uuid_type& other) const { return !(*this == other); }
    friend std::ostream& operator<<(std::ostream& out, const uuid_type& id)
    {
        out << id.to_string();
        return out;
    }

private:
    uint64_t m_data[2];
};
