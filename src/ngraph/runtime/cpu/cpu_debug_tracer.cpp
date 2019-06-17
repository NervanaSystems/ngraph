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

#include "ngraph/runtime/cpu/cpu_debug_tracer.hpp"
#include <chrono>

using namespace std;
using namespace ngraph;

runtime::cpu::CPU_DebugTracer::CPU_DebugTracer()
    : m_serial_number(0)
{
}

runtime::cpu::CPU_DebugTracer& runtime::cpu::CPU_DebugTracer::getInstance()
{
    static CPU_DebugTracer instance;
    return instance;
}

void runtime::cpu::CPU_DebugTracer::init_streams(const char* trace_file_path,
                                                 const char* trace_bin_file_path)
{
    if (m_tracer_stream.is_open())
    {
        return;
    }
    m_tracer_stream.open(trace_file_path, ios_base::out | ios_base::ate);
    m_tracer_bin_stream.open(trace_bin_file_path, std::ios_base::out | std::ios_base::ate);
}

void runtime::cpu::CPU_DebugTracer::end_of_kernel()
{
    m_serial_number++;

    m_tracer_stream.flush();
    m_tracer_bin_stream.flush();
}

// use of kahan sum to reduce numeric error
static float find_variance(const vector<float>& f_data, float mean, size_t size)
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

void runtime::cpu::CPU_DebugTracer::dump_one_tensor(
    const string& kernel_name,
    const ngraph::runtime::cpu::TensorViewWrapper& tv,
    const void* tensor,
    const string& tensor_name,
    const string& in_out)
{
    const ngraph::Shape& shape{tv.get_shape()};
    const size_t size = tv.get_size();

    float mean;
    float var;

    string tid{tensor_name.substr(1 + tensor_name.find("_"))};
    size_t num_bytes{(size * sizeof(float))};

    vector<float> float_data(size);

    memcpy(&float_data[0], tensor, num_bytes);

    m_tracer_stream << " K=" << left << setw(20) << kernel_name << " S=" << left << setw(10)
                    << m_serial_number << " TID=" << left << setw(10) << tid << in_out;

    m_tracer_bin_stream << "TID=" << tid << '\n';

    m_tracer_stream << " size=" << size << " " << shape << " ";

    m_tracer_stream << " bin_data_offset=" << m_tracer_bin_stream.tellp();
    m_tracer_bin_stream.write(reinterpret_cast<const char*>(float_data.data()),
                              float_data.size() * sizeof(float));

    mean = std::accumulate(float_data.begin(), float_data.end(), 0.0f) / size;

    var = find_variance(float_data, mean, size);

    m_tracer_stream << " mean=" << mean;
    m_tracer_stream << " var=" << var;

    m_tracer_bin_stream << "\n";
    m_tracer_stream << "\n";
}
