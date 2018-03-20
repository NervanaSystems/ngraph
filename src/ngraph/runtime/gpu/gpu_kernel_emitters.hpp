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

#include "ngraph/codegen/code_writer.hpp"
#include "ngraph/node.hpp"
#include "ngraph/runtime/gpu/gpu_tensor_view_wrapper.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            namespace kernel
            {
                void emit_memset(codegen::CodeWriter& writer,
                                 const GPU_TensorViewWrapper& dst,
                                 int value,
                                 size_t buffer_size = 0)
                {
                    if (buffer_size == 0)
                    {
                        buffer_size = dst.get_size() * dst.get_element_type().size();
                    }
                    writer << "runtime::gpu::cuda_memset(" << dst.get_name() << ", " << value
                           << ", " << buffer_size << ");\n";
                }

                void emit_memcpyDtD(codegen::CodeWriter& writer,
                                    const GPU_TensorViewWrapper& dst,
                                    const GPU_TensorViewWrapper& src)
                {
                    writer << "runtime::gpu::cuda_memcpyDtD(" << dst.get_name() << ", "
                           << src.get_name() << ", " << dst.get_size() << " * "
                           << dst.get_element_type().size() << ");\n";
                    return;
                }

                void emit_cudnnTensor4dDescriptor(codegen::CodeWriter& writer,
                                                  const std::string& name,
                                                  const std::string& format,
                                                  const std::string& data_type,
                                                  const std::array<size_t, 4>& axes)
                {
                    writer << "cudnnTensorDescriptor_t " << name << ";\n";
                    writer << "cudnnCreateTensorDescriptor(&" << name << ");\n";
                    writer << "cudnnSetTensor4dDescriptor(" << name << ",\n";
                    writer << "                 /*format=*/" << format << ",\n";
                    writer << "                 /*dataType=*/" << data_type;
                    for (auto const& axis : axes)
                    {
                        writer << ",\n                 /*dimension_size*/" << axis;
                    }
                    writer << ");\n";
                }

                void emit_cudnnTensorNdDescriptor(codegen::CodeWriter& writer,
                                                  const std::string& name,
                                                  const std::string& data_type,
                                                  const size_t& num_axes,
                                                  const std::vector<size_t>& axes,
                                                  const std::vector<size_t>& strides)
                {
                    writer << "const int " << name << "_axes[] = {";
                    for (auto const& axis : axes)
                    {
                        writer << axis << ",";
                    }
                    writer << "};\n";

                    writer << "const int " << name << "_strides[] = {";
                    for (auto const& axis_stride : strides)
                    {
                        writer << axis_stride << ",";
                    }
                    writer << "};\n";

                    writer << "cudnnTensorDescriptor_t " << name << ";\n";
                    writer << "cudnnCreateTensorDescriptor(&" << name << ");\n";
                    writer << "cudnnSetTensorNdDescriptor(" << name << ",\n";
                    writer << "                 /*dataType=*/" << data_type << ",\n";
                    writer << "                 /*num_dimensions=*/" << num_axes << ",\n";
                    writer << "                 /*dimensions*/" << name << "_axes,\n";
                    writer << "                 /*strides*/" << name << "_strides);\n";
                }

                void emit_cudnnReduceTensor(codegen::CodeWriter& writer,
                                            const GPU_TensorViewWrapper& in,
                                            const GPU_TensorViewWrapper& out,
                                            const std::string& reduce_op,
                                            const std::string& data_type,
                                            const std::string& nan_prop,
                                            const std::string& input_desc,
                                            const std::string& output_desc,
                                            const float& alpha,
                                            const float& beta)
                {
                    writer << "cudnnReduceTensorDescriptor_t reduceTensorDesc;\n";
                    writer << "cudnnCreateReduceTensorDescriptor(&reduceTensorDesc);\n";
                    writer << "cudnnSetReduceTensorDescriptor(reduceTensorDesc,\n";
                    writer << "                               " << reduce_op << ",\n";
                    writer << "                               " << data_type << ",\n";
                    writer << "                               " << nan_prop << ",\n";
                    writer << "                               CUDNN_REDUCE_TENSOR_NO_INDICES,\n";
                    writer << "                               CUDNN_32BIT_INDICES);\n";
                    writer << "size_t workspace_size = 0;\n";
                    writer << "cudnnGetReductionWorkspaceSize(cudnn_handle,\n";
                    writer << "                               reduceTensorDesc,\n";
                    writer << "                               " << input_desc << ",\n";
                    writer << "                               " << output_desc << ",\n";
                    writer << "                                &workspace_size);\n";
                    writer << "void* workspace_ptr = "
                              "ngraph::runtime::gpu::create_gpu_buffer(workspace_size);\n";
                    writer << "float alpha = " << alpha << ", beta = " << beta << ";\n";
                    writer << "cudnnReduceTensor(cudnn_handle,\n";
                    writer << "                  reduceTensorDesc,\n";
                    writer << "                  nullptr,\n";
                    writer << "                  0,\n";
                    writer << "                  workspace_ptr,\n";
                    writer << "                  workspace_size,\n";
                    writer << "                  &alpha,\n";
                    writer << "                  " << input_desc << ",\n";
                    writer << "                  " << in.get_name() << ",\n";
                    writer << "                  &beta,\n";
                    writer << "                  " << output_desc << ",\n";
                    writer << "                  " << out.get_name() << ");\n";
                    writer << "ngraph::runtime::gpu::free_gpu_buffer(workspace_ptr);\n";
                }
            }
        }
    }
}
