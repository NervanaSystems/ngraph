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

#include <chrono>
#include <cstdint>

#define TBB_PREVIEW_GLOBAL_CONTROL 1
#include <tbb/flow_graph.h>
#include <tbb/global_control.h>
#include <tbb/task_scheduler_init.h>

#ifdef NGRAPH_DISTRIBUTED
#include <mlsl.hpp>
#endif

namespace mkldnn
{
    class primitive;
}

namespace ngraph
{
    namespace runtime
    {
        class AlignedBuffer;
    }
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
                bool* p_en;
                bool first_iteration;
                mkldnn::primitive* const* mkldnn_primitives;
                std::vector<AlignedBuffer*> memory_buffers;
                char* const* mkldnn_workspaces;
                tbb::flow::graph* G;
                tbb::global_control* c;
                tbb::task_scheduler_init* init;
#ifdef NGRAPH_DISTRIBUTED
                MLSL::Environment* mlsl_env;
                MLSL::Distribution* mlsl_dist;
#endif
            };
            }
        }
    }
}
