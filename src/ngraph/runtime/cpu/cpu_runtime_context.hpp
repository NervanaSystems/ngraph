// ----------------------------------------------------------------------------
// Copyright 2018 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#pragma once

#include <chrono>
#include <cstdint>

namespace mkldnn
{
    class primitive;
}

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            typedef std::chrono::high_resolution_clock Clock;
            typedef std::chrono::time_point<Clock> Timestamp;
            typedef std::chrono::microseconds Timescale;

            extern "C" {
            struct CPURuntimeContext
            {
                int64_t* op_durations;
                mkldnn::primitive* const* mkldnn_primitives;
                char* const* mkldnn_workspaces;
            };
            }
        }
    }
}
