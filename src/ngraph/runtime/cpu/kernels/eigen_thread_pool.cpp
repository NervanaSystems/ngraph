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

#include "eigen_thread_pool.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace eigen
            {
                Eigen::ThreadPool global_thread_pool(Eigen::nbThreads());
                Eigen::ThreadPoolDevice global_thread_pool_device(&global_thread_pool,
                                                                  Eigen::nbThreads());
            }
        }
    }
}
