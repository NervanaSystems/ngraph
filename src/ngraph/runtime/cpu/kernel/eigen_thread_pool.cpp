/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <thread>

#include "eigen_thread_pool.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace eigen
            {
                static int GetNumCores()
                {
                    const auto omp_num_threads = std::getenv("OMP_NUM_THREADS");
                    const auto ngraph_intra_op_parallelism =
                        std::getenv("NGRAPH_INTRA_OP_PARALLELISM");
                    int count = 0;

                    if (omp_num_threads && (count = std::atoi(omp_num_threads)))
                    {
                        return count;
                    }
                    else if (ngraph_intra_op_parallelism &&
                             (count == std::atoi(ngraph_intra_op_parallelism)))
                    {
                        return count;
                    }
                    else
                    {
                        count = std::thread::hardware_concurrency() >> 1;
                    }
                    return count ? count : 1;
                }

                Eigen::ThreadPool global_thread_pool(GetNumCores());
                Eigen::ThreadPoolDevice global_thread_pool_device(&global_thread_pool,
                                                                  global_thread_pool.NumThreads());
            }
        }
    }
}
