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

#include <thread>

#include "cpu_executor.hpp"

#include "ngraph/except.hpp"

#define MAX_PARALLELISM_THRESHOLD 2

static int GetNumCores()
{
    const auto omp_num_threads = std::getenv("OMP_NUM_THREADS");
    const auto ngraph_intra_op_parallelism = std::getenv("NGRAPH_INTRA_OP_PARALLELISM");
    int count = 0;

    if (omp_num_threads)
    {
        count = std::atoi(omp_num_threads);
    }
    else if (ngraph_intra_op_parallelism)
    {
        count = std::atoi(ngraph_intra_op_parallelism);
    }
    else
    {
        count = std::thread::hardware_concurrency() / 2;
    }

    int max_parallelism_allowed = MAX_PARALLELISM_THRESHOLD * std::thread::hardware_concurrency();
    if (count > max_parallelism_allowed)
    {
        throw ngraph::ngraph_error(
            "OMP_NUM_THREADS and/or NGRAPH_INTRA_OP_PARALLELISM is too high: "
            "(" +
            std::to_string(count) + "). Please specify a value in range [1-" +
            std::to_string(max_parallelism_allowed) + "]");
    }

    return count < 1 ? 1 : count;
}

static int GetNumThreadPools()
{
    const auto ngraph_inter_op_parallelism = std::getenv("NGRAPH_INTER_OP_PARALLELISM");
    int count = 0;

    if (ngraph_inter_op_parallelism)
    {
        count = std::atoi(ngraph_inter_op_parallelism);
    }

    return count < 1 ? 1 : count;
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
                    m_num_cores = GetNumCores();
                    for (int i = 0; i < num_thread_pools; i++)
                    {
                        int num_threads_per_pool;

                        // Eigen threadpool will still be used for reductions
                        // and other tensor operations that dont use a parallelFor
                        num_threads_per_pool = GetNumCores();

                        // User override
                        char* eigen_tp_count = std::getenv("NGRAPH_CPU_EIGEN_THREAD_COUNT");
                        if (eigen_tp_count != nullptr)
                        {
                            const int tp_count = std::atoi(eigen_tp_count);
                            if (tp_count < 1 || tp_count > GetNumCores())
                            {
                                throw ngraph_error(
                                    "Unexpected value specified for NGRAPH_CPU_EIGEN_THREAD_COUNT "
                                    "(" +
                                    std::string(eigen_tp_count) +
                                    "). Please specify a value in range [1-" +
                                    std::to_string(GetNumCores()) + "]");
                            }
                            num_threads_per_pool = tp_count;
                        }

                        m_thread_pools.push_back(std::unique_ptr<Eigen::ThreadPool>(
                            new Eigen::ThreadPool(num_threads_per_pool)));
                        m_thread_pool_devices.push_back(
                            std::unique_ptr<Eigen::ThreadPoolDevice>(new Eigen::ThreadPoolDevice(
                                m_thread_pools[i].get(), num_threads_per_pool)));
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
                    static int num_thread_pools = GetNumThreadPools();
                    static CPUExecutor cpu_executor(num_thread_pools < 1 ? 1 : num_thread_pools);
                    return cpu_executor;
                }
#if MKLDNN_VERSION_MAJOR < 1
                mkldnn::engine global_cpu_engine(mkldnn::engine::cpu, 0);
#else
                mkldnn::engine global_cpu_engine(mkldnn::engine::kind::cpu, 0);
#endif
            }
        }
    }
}
