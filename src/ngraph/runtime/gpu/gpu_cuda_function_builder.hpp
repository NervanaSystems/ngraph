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

#include "ngraph/runtime/gpu/gpu_cuda_context_manager.hpp"
#include "ngraph/runtime/gpu/gpu_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            class Cuda_function_builder
            {
            public:
                static std::shared_ptr<CUfunction> Get(std::string& kernel,
                                                      std::string& name,
                                                      int number_of_options,
                                                      const char** options)
                {
                    nvrtcProgram prog;
                    NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog,
                                                       kernel.c_str(),
                                                       "op.cu",
                                                       0,      // numHeaders
                                                       NULL,   // headers
                                                       NULL)); // includeNames

                    nvrtcResult compileResult =
                        nvrtcCompileProgram(prog, number_of_options, options);

                    if (compileResult != NVRTC_SUCCESS)
                    {
                        // size_t logSize;
                        // NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
                        // char *log = new char[logSize];
                        // NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
                        // std::cout << log << '\n';
                        // delete[] log;
                        throw std::runtime_error("compile error: \n" + kernel + "\n options");
                    }

                    size_t ptxSize;
                    NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
                    char* ptx = new char[ptxSize];
                    NVRTC_SAFE_CALL(nvrtcGetPTX(
                        prog,
                        ptx)); // Load the generated PTX and get a handle to the parent kernel.
                    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog)); // Destroy the program.

                    CUmodule module;
                    CUfunction function;
                    CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
                    CUDA_SAFE_CALL(cuModuleGetFunction(&function, module, name.c_str()));
                    return std::make_shared<CUfunction>(function);
                }
            };
        }
    }
}
