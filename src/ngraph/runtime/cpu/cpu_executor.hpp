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

#include <thread>

#include <mkldnn.hpp>

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>

#define NGRAPH_CPU_THREAD_POOLS 1

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace executor
            {
                extern Eigen::ThreadPool global_thread_pool;
                extern Eigen::ThreadPoolDevice global_thread_pool_device;
                extern mkldnn::engine global_cpu_engine;

                // CPUExecutor owns the resources for executing a graph.
                class CPUExecutor
                {
                public:
                    CPUExecutor(int num_thread_pools);

                    int get_available_cores();

                    Eigen::ThreadPoolDevice& get_device(int id)
                    {
                        return *m_thread_pool_devices[id].get();
                    }

                private:
                    std::vector<std::unique_ptr<Eigen::ThreadPool>> m_thread_pools;
                    std::vector<std::unique_ptr<Eigen::ThreadPoolDevice>> m_thread_pool_devices;
                    int m_num_thread_pools;
                };

                extern CPUExecutor& GetCPUExecutor();
            }
        }
    }
}
