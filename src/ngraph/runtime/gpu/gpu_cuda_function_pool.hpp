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

#pragma once

#include <string>

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            class Cuda_function_pool
            {
                public:
                static Cuda_function_pool& Instance()
                {
                    static Cuda_function_pool pool;
                    return pool;
                }

                Cuda_function_pool(Cuda_function_pool const&) = delete;
                Cuda_function_pool(Cuda_function_pool&&) = delete;
                Cuda_function_pool& operator=(Cuda_function_pool const&) = delete;
                Cuda_function_pool& operator=(Cuda_function_pool &&) = delete;

                void Set(std::string& name, std::shared_ptr<CUfunction> function)
                {
                    CUfunction_map.insert({name,function});
                }

                std::shared_ptr<CUfunction> Get(std::string& name)
                {
                    auto it = CUfunction_map.find(name);
                    if(it != CUfunction_map.end())
                    {
                        return (*it).second;
                    }
                    return nullptr;
                }

                protected:
                Cuda_function_pool(){}
                ~Cuda_function_pool(){}

                std::unordered_map<std::string, std::shared_ptr<CUfunction>> CUfunction_map;
            }
        }
    }
}
