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

#include <functional>
#include <thread>

#include <mkldnn.hpp>

#include "ngraph/runtime/cpu/cpu_runtime_context.hpp"

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>

#if defined(NGRAPH_TBB_ENABLE)
#include "tbb/task_arena.h"
#endif

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace executor
            {
                extern mkldnn::engine global_cpu_engine;

                // CPUExecutor owns the resources for executing a graph.
                class CPUExecutor
                {
                public:
                    explicit CPUExecutor(int num_thread_pools);

                    Eigen::ThreadPoolDevice& get_device(int id)
                    {
                        return *m_thread_pool_devices[id].get();
                    }

#if defined(NGRAPH_TBB_ENABLE)
                    void execute(CPUKernelFunctor& f,
                                 CPURuntimeContext* ctx,
                                 CPUExecutionContext* ectx,
                                 bool use_tbb = false);
#else
                    void execute(CPUKernelFunctor& f,
                                 CPURuntimeContext* ctx,
                                 CPUExecutionContext* ectx);
#endif
                    int get_num_thread_pools() { return m_num_thread_pools; }
                    int get_num_cores() { return m_num_cores; }
                private:
                    std::vector<std::unique_ptr<Eigen::ThreadPool>> m_thread_pools;
                    std::vector<std::unique_ptr<Eigen::ThreadPoolDevice>> m_thread_pool_devices;
#if defined(NGRAPH_TBB_ENABLE)
                    std::vector<tbb::task_arena> m_tbb_arenas;
#endif
                    int m_num_thread_pools;
                    int m_num_cores;
                };

                extern CPUExecutor& GetCPUExecutor();
            }
        }
    }
}
