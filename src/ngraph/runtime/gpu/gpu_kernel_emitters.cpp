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

#include <algorithm>
#include <map>

#include "gpu_kernel_emitters.hpp"
#include "ngraph/codegen/code_writer.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;

void runtime::gpu::kernel::emit_memset(codegen::CodeWriter& writer,
                                       const GPU_TensorViewWrapper& dst,
                                       int value,
                                       size_t buffer_size)
{
    if (buffer_size == 0)
    {
        buffer_size = dst.get_size() * dst.get_element_type().size();
    }
    writer << "runtime::gpu::cuda_memset(" << dst.get_name() << ", " << value << ", " << buffer_size
           << ");\n";
}

void runtime::gpu::kernel::emit_memcpyDtD(codegen::CodeWriter& writer,
                                          const GPU_TensorViewWrapper& dst,
                                          const GPU_TensorViewWrapper& src)
{
    writer << "runtime::gpu::cuda_memcpyDtD(" << dst.get_name() << ", " << src.get_name() << ", "
           << dst.get_size() << " * " << dst.get_element_type().size() << ");\n";
    return;
}

void emit_cudnnConvolutionDescriptor(codegen::CodeWriter& writer,
                                    const std::string& name,
                                    const CoordinateDiff& padding,
                                    const Strides& window_movement_strides,
                                    const Strides& window_dilation_strides,
                                    const std::string& mode,
                                    const std::string& data_type);
{
    writer << "cudnnConvolutionDescriptor_t " << name << ";\n";
    writer << "cudnnCreateConvolutionDescriptor(&" << name << ");\n";

    if(padding.size() == 2)
    {
        writer << "cudnnSetConvolution2dDescriptor("
                << name << ", "
                << padding[0] << ", "
                << padding[1] << ", "
                << window_movement_strides[0] << ", "
                << window_movement_strides[1] << ", "
                << window_dilation_strides[0] << ", "
                << window_dilation_strides[1] << ", "
                << mode <<", "
                << data_type
                << ");\n";
    }
    else
    {
        writer << "cudnnSetConvolutionNdDescriptor("
                << name << ", "
                << padding.size() << ", "
                << "{" << join(padding) << "}, "
                << "{" << join(window_movement_strides) << "}, "
                << "{" << join(window_dilation_strides) << "}, "
                << mode <<", "
                << data_type
                << ");\n";
    }

}

void runtime::gpu::kernel::emit_cudnnTensorDescriptor(codegen::CodeWriter& writer,
                                                        const std::string& name,
                                                        const std::string& format,
                                                        const std::string& data_type,
                                                        const Shape& shape)
{
    std::vector<size_t> dimensions;
    for (size_t i = shape.size(); i < 4; i++)
    {
        dimensions.push_back(1);
    }
    for (size_t i = 0; i < shape.size(); i++)
    {
        dimensions.push_back(shape[i]);
    }

    writer << "cudnnTensorDescriptor_t " << name << ";\n";
    writer << "cudnnCreateTensorDescriptor(&" << name << ");\n";

    if(dimensions.size() < 4)
    {
        writer << "cudnnSetTensor4dDescriptor(" << name << ",\n";
        writer << "                 /*format=*/" << format << ",\n";
        writer << "                 /*dataType=*/" << data_type;
        writer << ",\n                 /*dimension_size*/" << dimensions[0];
        writer << ",\n                 /*dimension_size*/" << dimensions[1];
        writer << ",\n                 /*dimension_size*/" << dimensions[2];
        writer << ",\n                 /*dimension_size*/" << dimensions[3];
        writer << ");\n";
    }
    else
    {
        auto compute_strides = [](const std::vector<size_t>& dim) {
            std::vector<size_t> strides(dim.size(), 1);
            std::copy(dim.begin() + 1, dim.end(), strides.begin());
            for (int64_t i = dim.size() - 2; i >= 0; i--)
            {
                strides[i] *= strides[i + 1];
            }
            return strides;
        };
        std::vector<size_t> strides = compute_strides(dimensions);
        writer << "const int " << name << "_axes[] = {" << join(dimensions) << "};\n";
        writer << "const int " << name << "_strides[] = {" << join(strides) << "};\n";
        writer << "cudnnSetTensorNdDescriptor(" << name << ",\n";
        writer << "                 /*dataType=*/" << data_type << ",\n";
        writer << "                 /*num_dimensions=*/" << dimensions.size() << ",\n";
        writer << "                 /*dimensions*/" << name << "_axes,\n";
        writer << "                 /*strides*/" << name << "_strides);\n";
    }
}

void runtime::gpu::kernel::emit_cudnnFilterDescriptor(codegen::CodeWriter& writer,
                                                        const std::string& name,
                                                        const std::string& format,
                                                        const std::string& data_type,
                                                        const Shape& shape)
{
    std::vector<size_t> dimensions;
    for (size_t i = shape.size(); i < 4; i++)
    {
        dimensions.push_back(1);
    }
    for (size_t i = 0; i < shape.size(); i++)
    {
        dimensions.push_back(shape[i]);
    }

    writer << "cudnnFilterDescriptor_t " << name << ";\n";
    writer << "cudnnCreateFilterDescriptor(&" << name << ");\n";

    if(dimensions.size() < 4)
    {
        writer << "cudnnSetFilter4dDescriptor(" << name << ",\n";
        writer << "                 /*dataType=*/" << data_type << ",\n";
        writer << "                 /*format=*/" << format;
        writer << ",\n                 /*dimension_size*/" << dimensions[0];
        writer << ",\n                 /*dimension_size*/" << dimensions[1];
        writer << ",\n                 /*dimension_size*/" << dimensions[2];
        writer << ",\n                 /*dimension_size*/" << dimensions[3];
        writer << ");\n";
    }
    else
    {
        writer << "const int " << name << "_axes[] = {" << join(dimensions) << "};\n";
        writer << "cudnnSetFilterNdDescriptor(" << name << ",\n";
        writer << "                 /*dataType=*/" << data_type << ",\n";
        writer << "                 /*format=*/" << format << ",\n";
        writer << "                 /*num_dimensions=*/" << dimensions.size() << ",\n";
        writer << "                 /*dimensions*/" << name << "_axes);\n";
    }
}


void runtime::gpu::kernel::emit_cudnnTensor4dDescriptor(codegen::CodeWriter& writer,
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

void runtime::gpu::kernel::emit_cudnnTensorNdDescriptor(codegen::CodeWriter& writer,
                                                        const std::string& name,
                                                        const std::string& data_type,
                                                        const size_t& num_axes,
                                                        const std::vector<size_t>& axes,
                                                        const std::vector<size_t>& strides)
{
    writer << "const int " << name << "_axes[] = {" << join(axes) << "};\n";
    writer << "const int " << name << "_strides[] = {" << join(strides) << "};\n";
    writer << "cudnnTensorDescriptor_t " << name << ";\n";
    writer << "cudnnCreateTensorDescriptor(&" << name << ");\n";
    writer << "cudnnSetTensorNdDescriptor(" << name << ",\n";
    writer << "                 /*dataType=*/" << data_type << ",\n";
    writer << "                 /*num_dimensions=*/" << num_axes << ",\n";
    writer << "                 /*dimensions*/" << name << "_axes,\n";
    writer << "                 /*strides*/" << name << "_strides);\n";
}

void runtime::gpu::kernel::emit_cudnnReduceTensor(codegen::CodeWriter& writer,
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
