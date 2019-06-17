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

#include "ngraph/runtime/cpu/cpu_tensor_view_wrapper.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            class CPU_DebugTracer
            {
            public:
                CPU_DebugTracer(const CPU_DebugTracer&) = delete;
                CPU_DebugTracer& operator=(const CPU_DebugTracer&) = delete;

                static CPU_DebugTracer& getInstance();

                void init_streams(const char*, const char*);

                void end_of_kernel();

                void dump_one_tensor(const std::string& kernel_name,
                                     const ngraph::runtime::cpu::TensorViewWrapper& tv,
                                     const void* tensor,
                                     const std::string& tensor_name,
                                     const std::string& in_out);

            private:
                CPU_DebugTracer();

                size_t m_serial_number;
                std::fstream m_tracer_stream;
                std::fstream m_tracer_bin_stream;
            };
        }
    }
}