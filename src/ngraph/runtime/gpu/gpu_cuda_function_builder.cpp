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

#include <cstring>
#include <iostream>
#include <string>

#include "ngraph/runtime/gpu/gpu_cuda_context_manager.hpp"
#include "ngraph/runtime/gpu/gpu_cuda_function_builder.hpp"
#include "ngraph/runtime/gpu/gpu_util.hpp"

using namespace ngraph;

std::shared_ptr<CUfunction> runtime::gpu::CudaFunctionBuilder::get(const std::string& name,
                                                                   const std::string& kernel,
                                                                   int number_of_options,
                                                                   const char** options)
{
    nvrtcProgram prog;
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog,
                                       kernel.c_str(),
                                       "ngraph.cu",
                                       0,      // numHeaders
                                       NULL,   // headers
                                       NULL)); // includeNames

    nvrtcResult compile_result = nvrtcCompileProgram(prog, number_of_options, options);

    // output compiler log helper
    auto emit_log = [&prog]() {
        size_t logSize;
        NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
        char* log = static_cast<char*>(malloc(sizeof(char) * logSize + 1));
        NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
        log[logSize] = '\x0';
        if (std::strlen(log) >= 2)
        {
            std::cerr << log;
        }
        free(log);
    };

    // throw if compilation was not successful
    if (compile_result != NVRTC_SUCCESS)
    {
        std::cerr << "Compile error: \n" + kernel;
        // output compiler errors
        emit_log();
        throw std::runtime_error("NVRTC compilation failure.");
    }
    // output any compiler warnings
    emit_log();

    // retrieve the intermediate PTX
    size_t ptx_size;
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptx_size));
    char* ptx = new char[ptx_size];
    NVRTC_SAFE_CALL(
        nvrtcGetPTX(prog,
                    ptx)); // Load the generated PTX and get a handle to the parent kernel.
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog)); // Destroy the program.

    // extract the compiled function
    CUmodule module;
    CUfunction function;
    CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
    CUDA_SAFE_CALL(cuModuleGetFunction(&function, module, name.c_str()));
    return std::make_shared<CUfunction>(function);
}
