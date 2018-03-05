/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include <string>
#include <unordered_map>

#include "ngraph/runtime/gpu/gpu_cuda_function_pool.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            CudaFunctionPool& CudaFunctionPool::instance()
            {
                static CudaFunctionPool pool;
                return pool;
            }

            void CudaFunctionPool::set(std::string& name, std::shared_ptr<CUfunction> function)
            {
                m_function_map.insert({name, function});
            }

            std::shared_ptr<CUfunction> CudaFunctionPool::get(std::string& name)
            {
                auto it = m_function_map.find(name);
                if (it != m_function_map.end())
                {
                    return (*it).second;
                }
                return nullptr;
            }
        }
    }
}
