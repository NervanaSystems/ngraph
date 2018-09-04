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

#include <string>
#include <unordered_map>

#include "ngraph/runtime/gpu/gpu_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            class CudaFunctionPool
            {
            public:
                CudaFunctionPool() {}
                ~CudaFunctionPool() {}
                std::shared_ptr<CUfunction> set(const std::string& name, const std::string& kernel);
                std::shared_ptr<CUfunction> get(const std::string& name);

            private:
                std::unordered_map<std::string, std::shared_ptr<CUfunction>> m_function_map;
            };
        }
    }
}
