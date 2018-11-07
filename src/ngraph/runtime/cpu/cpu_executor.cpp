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

#include <thread>

#include "cpu_executor.hpp"

static int GetNumCores()
{
    const auto omp_num_threads = std::getenv("OMP_NUM_THREADS");
    const auto ngraph_intra_op_parallelism = std::getenv("NGRAPH_INTRA_OP_PARALLELISM");
    int count = 0;

    if (omp_num_threads && (count = std::atoi(omp_num_threads)))
    {
        return count;
    }
    else if (ngraph_intra_op_parallelism && (count = std::atoi(ngraph_intra_op_parallelism)))
    {
        return count;
    }
    else
    {
        count = std::thread::hardware_concurrency() >> 1;
    }
    return count ? count : 1;
}

static int GetNumThreadPools()
{
    const auto ngraph_inter_op_parallelism = std::getenv("NGRAPH_INTER_OP_PARALLELISM");
    int count = 0;

    if (ngraph_inter_op_parallelism && (count = std::atoi(ngraph_inter_op_parallelism)))
    {
        return count;
    }

    return 1;
}

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace executor
            {
                CPUExecutor::CPUExecutor(int num_thread_pools)
                    : m_num_thread_pools(num_thread_pools)
                {
                    for (int i = 0; i < num_thread_pools; i++)
                    {
                        int num_threads_per_pool;
#if defined(EIGEN_OPENMP)
                        num_threads_per_pool = 1;
#else
                        num_threads_per_pool = GetNumCores();
#endif
                        m_thread_pools.push_back(std::unique_ptr<Eigen::ThreadPool>(
                            new Eigen::ThreadPool(num_threads_per_pool)));
                        m_thread_pool_devices.push_back(std::unique_ptr<Eigen::ThreadPoolDevice>(
                            new Eigen::ThreadPoolDevice(m_thread_pools[i].get(), GetNumCores())));
                        m_tbb_arenas.emplace_back(1);
                    }
                }

                void CPUExecutor::execute(CPUKernelFunctor& f,
                                          CPURuntimeContext* ctx,
                                          CPUExecutionContext* ectx,
                                          bool use_tbb)
                {
                    auto tbb_functor = [&]() { f(ctx, ectx); };
                    if (use_tbb)
                    {
                        m_tbb_arenas[ectx->arena].execute(tbb_functor);
                    }
                    else
                    {
                        f(ctx, ectx);
                    }
                }

                CPUExecutor& GetCPUExecutor()
                {
                    static CPUExecutor cpu_executor(GetNumThreadPools());
                    return cpu_executor;
                }

                mkldnn::engine global_cpu_engine(mkldnn::engine::cpu, 0);
            }
        }
    }
}
