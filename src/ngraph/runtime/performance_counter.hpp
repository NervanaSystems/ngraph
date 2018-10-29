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

#include <cstddef>
#include <string>

namespace ngraph
{
    namespace runtime
    {
        class PerformanceCounter
        {
        public:
            PerformanceCounter(const char* n, size_t us, size_t calls)
                : m_name(n)
                , m_total_microseconds(us)
                , m_call_count(calls)
            {
            }
            const std::string& name() const { return m_name; }
            size_t total_microseconds() const { return m_total_microseconds; }
            size_t microseconds() const
            {
                return m_call_count == 0 ? 0 : m_total_microseconds / m_call_count;
            }
            size_t call_count() const { return m_call_count; }
            std::string m_name;
            size_t m_total_microseconds;
            size_t m_call_count;
        };
    }
}
