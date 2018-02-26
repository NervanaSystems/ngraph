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
            class CudaKernelBuilder
            {
            public:
                static void get_1_element_op(const std::string& name,
                                             const std::string& data_type,
                                             const std::string& op,
                                             std::string& kernel)
                {
                    kernel = R"(  
extern "C" __global__
void cuda_)" + name + "(" + data_type +
                             "* in, " + data_type + "* out, size_t n)\n" + R"({  
size_t tid = blockIdx.x * blockDim.x + threadIdx.x;  
if(tid < n) 
{
out[tid] =)" + op + "(in[tid]);\n" +
                             R"(}
})";
                    return;
                }

                static void get_2_element_op(const std::string& name,
                                             const std::string& data_type,
                                             const std::string& op,
                                             std::string& kernel)
                {
                    kernel = R"(  
extern "C" __global__
void )" + name + "(" + data_type +
                             "* in1, " + data_type + "* in2, " + data_type + "* out, size_t n)\n" +
                             R"({  
size_t tid = blockIdx.x * blockDim.x + threadIdx.x;  
if(tid < n) 
{
out[tid] = in1[tid] )" + op + "in2[tid]\n" +
                             R"(}
})";
                    return;
                }

                static void get_n_element_op(const std::string& name,
                                             const std::string& data_type,
                                             const std::vector<std::string>& ops,
                                             std::string& kernel)
                {
                    kernel = "";
                    return;
                }
            };
        }
    }
}
