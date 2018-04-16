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

#include <cctype>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include "ngraph/file_util.hpp"
#include "ngraph/runtime/gpu/gpu_cuda_function_builder.hpp"
#include "ngraph/runtime/gpu/gpu_cuda_function_pool.hpp"

static const std::string s_output_dir = "gpu_codegen";

using namespace ngraph;

std::shared_ptr<CUfunction> runtime::gpu::CudaFunctionPool::set(const std::string& name,
                                                                const std::string& kernel)
{
    const char* opts[] = {"--gpu-architecture=compute_35", "--relocatable-device-code=true"};
    std::string filename =
        file_util::path_join(s_output_dir, "cuda_kernel_" + name + "_codegen.cu");
    std::ofstream out(filename);
    out << kernel;
    out.close();
    auto cu_compiled_function = CudaFunctionBuilder::get("cuda_" + name, kernel, 2, opts);
    m_function_map.insert({name, cu_compiled_function});
    return cu_compiled_function;
}

std::shared_ptr<CUfunction> runtime::gpu::CudaFunctionPool::get(const std::string& name)
{
    auto it = m_function_map.find(name);
    if (it != m_function_map.end())
    {
        return (*it).second;
    }
    return nullptr;
}
