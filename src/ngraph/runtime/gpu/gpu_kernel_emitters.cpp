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
                                          const GPU_TensorViewWrapper& src,
                                          size_t buffer_size)
{
    if (buffer_size == 0)
    {
        writer << "runtime::gpu::cuda_memcpyDtD(" << dst.get_name() << ", " << src.get_name()
               << ", " << dst.get_size() << " * " << dst.get_element_type().size() << ");\n";
        return;
    }
    writer << "runtime::gpu::cuda_memcpyDtD(" << dst.get_name() << ", " << src.get_name() << ", "
           << buffer_size << ");\n";
    return;
}

void runtime::gpu::kernel::emit_cudnnConvolutionDescriptor(codegen::CodeWriter& writer,
                                                           const std::string& name,
                                                           const CoordinateDiff& padding,
                                                           const Strides& window_movement_strides,
                                                           const Strides& window_dilation_strides,
                                                           const std::string& mode,
                                                           const std::string& data_type)
{
    writer << "auto& " << name << " = descriptors.build<cudnnConvolutionDescriptor_t>();\n";

    if (padding.size() == 2)
    {
        writer << "CUDNN_SAFE_CALL(cudnnSetConvolution2dDescriptor(" << name << ", " << padding[0]
               << ", " << padding[1] << ", " << window_movement_strides[0] << ", "
               << window_movement_strides[1] << ", " << window_dilation_strides[0] << ", "
               << window_dilation_strides[1] << ", " << mode << ", " << data_type << "));\n";
    }
    else
    {
        writer << "const int " << name << "_padding[] = {" << join(padding) << "};\n";
        writer << "const int " << name << "_window_movement_strides[] = {"
               << join(window_movement_strides) << "};\n";
        writer << "const int " << name << "_window_dilation_strides[] = {"
               << join(window_dilation_strides) << "};\n";

        writer << "CUDNN_SAFE_CALL(cudnnSetConvolutionNdDescriptor(" << name << ", "
               << padding.size() << ", " << name << "_padding, " << name
               << "_window_movement_strides, " << name << "_window_dilation_strides, " << mode
               << ", " << data_type << "));\n";
    }
}

void runtime::gpu::kernel::emit_cudnnFilterDescriptor(codegen::CodeWriter& writer,
                                                      const std::string& name,
                                                      const std::string& format,
                                                      const std::string& data_type,
                                                      const Shape& shape)
{
    Shape dimensions(fmax(4, shape.size()), 1);
    int idx = 0;
    for (size_t i = dimensions.size() - shape.size(); i < dimensions.size(); i++)
    {
        dimensions[i] = shape[idx++];
    }

    writer << "auto& " << name << " = descriptors.build<cudnnFilterDescriptor_t>();\n";

    if (dimensions.size() <= 4)
    {
        writer << "CUDNN_SAFE_CALL(cudnnSetFilter4dDescriptor(" << name << ",\n";
        writer << "                 /*dataType=*/" << data_type << ",\n";
        writer << "                 /*format=*/" << format << ",\n";
        writer << "                 /*dimension_size*/" << dimensions[0] << ",\n";
        writer << "                 /*dimension_size*/" << dimensions[1] << ",\n";
        writer << "                 /*dimension_size*/" << dimensions[2] << ",\n";
        writer << "                 /*dimension_size*/" << dimensions[3] << "));\n";
    }
    else
    {
        writer << "const int " << name << "_axes[] = {" << join(dimensions) << "};\n";
        writer << "CUDNN_SAFE_CALL(cudnnSetFilterNdDescriptor(" << name << ",\n";
        writer << "                 /*dataType=*/" << data_type << ",\n";
        writer << "                 /*format=*/" << format << ",\n";
        writer << "                 /*num_dimensions=*/" << dimensions.size() << ",\n";
        writer << "                 /*dimensions*/" << name << "_axes));\n";
    }
}

void runtime::gpu::kernel::emit_cudnnTensorDescriptor(codegen::CodeWriter& writer,
                                                      const std::string& name,
                                                      const std::string& format,
                                                      const std::string& data_type,
                                                      const Shape& shape)
{
    Shape dimensions(fmax(4, shape.size()), 1);
    int idx = 0;
    for (size_t i = dimensions.size() - shape.size(); i < dimensions.size(); i++)
    {
        dimensions[i] = shape[idx++];
    }

    writer << "auto& " << name << " = descriptors.build<cudnnTensorDescriptor_t>();\n";
    if (dimensions.size() <= 4)
    {
        writer << "CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(" << name << ",\n";
        writer << "                 /*format=*/" << format << ",\n";
        writer << "                 /*dataType=*/" << data_type << ",\n";
        writer << "                 /*dimension_size*/" << dimensions[0] << ",\n";
        writer << "                 /*dimension_size*/" << dimensions[1] << ",\n";
        writer << "                 /*dimension_size*/" << dimensions[2] << ",\n";
        writer << "                 /*dimension_size*/" << dimensions[3] << "));\n";
    }
    else
    {
        Strides strides = row_major_strides(dimensions);
        writer << "const int " << name << "_axes[] = {" << join(dimensions) << "};\n";
        writer << "const int " << name << "_strides[] = {" << join(strides) << "};\n";
        writer << "CUDNN_SAFE_CALL(cudnnSetTensorNdDescriptor(" << name << ",\n";
        writer << "                 /*dataType=*/" << data_type << ",\n";
        writer << "                 /*num_dimensions=*/" << dimensions.size() << ",\n";
        writer << "                 /*dimensions*/" << name << "_axes,\n";
        writer << "                 /*strides*/" << name << "_strides));\n";
    }
}

void runtime::gpu::kernel::emit_cudnnTensor4dDescriptor(codegen::CodeWriter& writer,
                                                        const std::string& name,
                                                        const std::string& format,
                                                        const std::string& data_type,
                                                        const std::array<size_t, 4>& axes)
{
    writer << "auto& " << name << " = descriptors.build<cudnnTensorDescriptor_t>();\n";
    writer << "CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(" << name << ",\n";
    writer << "                 /*format=*/" << format << ",\n";
    writer << "                 /*dataType=*/" << data_type;
    for (auto const& axis : axes)
    {
        writer << ",\n                 /*dimension_size*/" << axis;
    }
    writer << "));\n";
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
    writer << "auto& " << name << " = descriptors.build<cudnnTensorDescriptor_t>();\n";
    writer << "CUDNN_SAFE_CALL(cudnnSetTensorNdDescriptor(" << name << ",\n";
    writer << "                 /*dataType=*/" << data_type << ",\n";
    writer << "                 /*num_dimensions=*/" << num_axes << ",\n";
    writer << "                 /*dimensions*/" << name << "_axes,\n";
    writer << "                 /*strides*/" << name << "_strides));\n";
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
    writer << "auto& reduceTensorDesc = descriptors.build<cudnnReduceTensorDescriptor_t>();\n";
    writer << "CUDNN_SAFE_CALL(cudnnSetReduceTensorDescriptor(reduceTensorDesc,\n";
    writer << "                               " << reduce_op << ",\n";
    writer << "                               " << data_type << ",\n";
    writer << "                               " << nan_prop << ",\n";
    writer << "                               CUDNN_REDUCE_TENSOR_NO_INDICES,\n";
    writer << "                               CUDNN_32BIT_INDICES));\n";
    writer << "size_t workspace_size = 0;\n";
    writer << "CUDNN_SAFE_CALL(cudnnGetReductionWorkspaceSize(*ctx->cudnn_handle,\n";
    writer << "                               reduceTensorDesc,\n";
    writer << "                               " << input_desc << ",\n";
    writer << "                               " << output_desc << ",\n";
    writer << "                                &workspace_size));\n";
    writer << "void* workspace_ptr = "
              "ngraph::runtime::gpu::create_gpu_buffer(workspace_size);\n";
    writer << "float alpha = " << alpha << ", beta = " << beta << ";\n";
    writer << "CUDNN_SAFE_CALL(cudnnReduceTensor(*ctx->cudnn_handle,\n";
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
    writer << "                  " << out.get_name() << "));\n";
    writer << "ngraph::runtime::gpu::free_gpu_buffer(workspace_ptr);\n";
}

std::string runtime::gpu::kernel::emit_type_string(const Node* node)
{
    std::stringstream ss;
    for (auto const& input : node->get_inputs())
    {
        ss << input.get_element_type().c_type_string() << "_";
    }
    for (auto const& output : node->get_outputs())
    {
        ss << output.get_element_type().c_type_string() << "_";
    }
    std::string types = ss.str();
    std::replace(types.begin(), types.end(), ' ', '_');
    return types;
}
