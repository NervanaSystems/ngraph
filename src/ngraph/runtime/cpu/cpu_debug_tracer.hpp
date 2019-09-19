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

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "ngraph/runtime/tensor.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            class CPU_DebugTracer
            {
            public:
                CPU_DebugTracer();

                void set_enable_tracing(bool new_state);

                bool tracing_is_enabled() { return m_enable_tracing; }
                void end_of_kernel();

                template <typename T>
                void dump_one_tensor(const std::string& kernel_name,
                                     const void* tensor,
                                     const std::string& tensor_name,
                                     const size_t size,
                                     const ngraph::Shape& shape,
                                     const std::string& in_out);

            private:
                CPU_DebugTracer(const CPU_DebugTracer&) = delete;
                CPU_DebugTracer(CPU_DebugTracer&&) = delete;
                CPU_DebugTracer& operator=(const CPU_DebugTracer&) = delete;

                void init_streams();

                size_t m_serial_number;
                std::fstream m_tracer_stream;
                std::fstream m_tracer_bin_stream;

                bool m_enable_tracing = false;
            };
        }
    }
}

// use of kahan sum to reduce numeric error
template <typename T>
float find_variance(const std::vector<T>& f_data, float mean, size_t size)
{
    float sum = 0.0f;
    float c = 0.0f;
    for (auto num : f_data)
    {
        num = (num - mean) * (num - mean);
        float y = num - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum / size;
}

template <typename T>
void ngraph::runtime::cpu::CPU_DebugTracer::dump_one_tensor(const std::string& kernel_name,
                                                            const void* tensor,
                                                            const std::string& tensor_name,
                                                            const size_t size,
                                                            const ngraph::Shape& shape,
                                                            const std::string& in_out)
{
    std::string tid{tensor_name.substr(1 + tensor_name.find("_"))};
    size_t num_bytes{(size * sizeof(T))};

    std::vector<T> tensor_data(size);

    memcpy(&tensor_data[0], tensor, num_bytes);

    m_tracer_stream << " K=" << std::left << std::setw(20) << kernel_name << " S=" << std::left
                    << std::setw(10) << m_serial_number << " TID=" << std::left << std::setw(10)
                    << tid << in_out;

    m_tracer_bin_stream << "TID=" << tid << '\n';

    m_tracer_stream << " size=" << size << " " << shape << " ";

    m_tracer_stream << "bin_data_offset=" << m_tracer_bin_stream.tellp();
    m_tracer_bin_stream.write(reinterpret_cast<const char*>(tensor_data.data()),
                              tensor_data.size() * sizeof(T));

    auto mean = std::accumulate(tensor_data.begin(), tensor_data.end(), 0.0f) / size;

    auto var = find_variance<T>(tensor_data, mean, size);

    m_tracer_stream << " mean=" << mean;
    m_tracer_stream << " var=" << var;

    m_tracer_bin_stream << "\n";
    m_tracer_stream << "\n";
}
