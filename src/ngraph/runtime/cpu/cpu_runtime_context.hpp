//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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
#include <set>

#if defined(NGRAPH_TBB_ENABLE)
#define TBB_PREVIEW_GLOBAL_CONTROL 1
#define TBB_PREVIEW_FLOW_GRAPH_TRACE 1
#include <tbb/flow_graph.h>
#include <tbb/global_control.h>
#endif

#include "ngraph/op/experimental/compiled_kernel.hpp"

#ifdef NGRAPH_IN_CODEGEN
#undef NGRAPH_CPU_MLIR_ENABLE
#endif

#ifdef NGRAPH_CPU_MLIR_ENABLE
#include "contrib/mlir/runtime/cpu/cpu_runtime.hpp"
#endif

namespace dnnl
{
    struct primitive;
}

namespace ngraph
{
    namespace runtime
    {
        class AlignedBuffer;
    }
    class State;
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
                // stores tensor pointers
                std::vector<void*> buffer_data;
                std::vector<dnnl::memory*> dnnl_memories;
                std::vector<dnnl::primitive*> dnnl_primitives;
                std::vector<AlignedBuffer*> memory_buffers;
                std::vector<dnnl::memory::desc*> dnnl_scratchpad_mds;
                AlignedBuffer* scratchpad_buffer;
                std::vector<char*> dnnl_workspaces;
#if defined(NGRAPH_TBB_ENABLE)
                tbb::flow::graph* G;
                tbb::global_control* c;
#endif
                State* const* states;
                std::set<size_t> breakpoints;
                size_t pc;
#ifdef NGRAPH_CPU_MLIR_ENABLE
                /// Maps CompiledKernel nodes to their MLIR compiler
                /// The MLIR compiler caches the compiled code on the first invocation,
                /// and may in the future support re-compilation
                std::unordered_map<ngraph::op::v0::CompiledKernel*,
                                   ngraph::runtime::ngmlir::MLIRCPURuntime>
                    mlir_runtimes;
#endif
            };
            }

            struct CPUExecutionContext
            {
                int arena;
            };

            typedef std::function<void(CPURuntimeContext*, CPUExecutionContext*)> CPUKernelFunctor;
        }
    }
}
