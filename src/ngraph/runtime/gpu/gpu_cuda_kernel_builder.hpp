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

#include <array>
#include <string>
#include <vector>

namespace ngraph
{
    namespace codegen
    {
        class CodeWriter;
    }
    namespace runtime
    {
        namespace gpu
        {
            class CudaKernelBuilder
            {
            public:
                static void get_elementwise_op(codegen::CodeWriter& writer,
                                               const std::string& name,
                                               const std::string& op,
                                               const std::array<std::string, 2>& data_types,
                                               const size_t& num_inputs);

                static void get_device_helper(codegen::CodeWriter& writer,
                                              const std::string& name,
                                              const std::string& math_kernel,
                                              const std::array<std::string, 2>& data_types,
                                              const size_t& num_inputs);
                static void add_pod_typedefs(codegen::CodeWriter& writer);
            };
        }
    }
}
