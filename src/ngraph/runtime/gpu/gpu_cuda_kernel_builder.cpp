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
#include "ngraph/runtime/gpu/gpu_cuda_kernel_builder.hpp"
#include "ngraph/codegen/code_writer.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            void CudaKernelBuilder::get_elementwise_op(codegen::CodeWriter& writer,
                                                       const std::string& name,
                                                       const std::string& data_type,
                                                       const std::string& op,
                                                       const size_t& num_inputs)
            {
                writer << "extern \"C\" __global__ void cuda_" << name << "(";
                for (size_t i = 0; i < num_inputs; i++)
                {
                    writer << data_type << "* in" << i << ", ";
                }
                writer << data_type << "* out,"
                       << "size_t n)\n";
                writer << "{\n";
                writer.indent++;
                {
                    writer << "size_t tid = blockIdx.x * blockDim.x + threadIdx.x; \n";
                    writer << "if (tid < n)\n";
                    writer << "{\n";
                    writer.indent++;
                    {
                        writer << "out[tid] = " << op << "(";
                        for (size_t i = 0; i < num_inputs - 1; i++)
                        {
                            writer << "in" << i << "[tid], ";
                        }
                        writer << "in" << num_inputs - 1 << "[tid]);\n";
                    }
                    writer.indent--;
                    writer << "}\n";
                }
                writer.indent--;
                writer << "}\n";

                return;
            }

            void CudaKernelBuilder::get_device_helper(codegen::CodeWriter& writer,
                                                      const std::string& name,
                                                      const std::string& data_type,
                                                      const std::string& math_kernel,
                                                      const size_t& num_inputs)
            {
                if (math_kernel.size())
                {
                    writer << "__device__ " << data_type << " " << name << "(";
                    for (size_t i = 0; i < num_inputs - 1; i++)
                    {
                        writer << data_type << " x" << i << ", ";
                    }
                    writer << data_type << " x" << num_inputs - 1;
                    writer << ")\n";
                    writer << "{\n";
                    writer.indent++;
                    {
                        writer << "return " + math_kernel << ";\n";
                    }
                    writer.indent--;
                    writer << "}\n";
                }
                return;
            }
        }
    }
}
