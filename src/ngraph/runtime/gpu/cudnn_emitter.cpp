/*******************************************************************************
* Copyright 2018 Intel Corporation
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
#include <iostream>
#include <vector>

#include "ngraph/runtime/gpu/cudnn_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_primitive_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_runtime_context.hpp"

using namespace ngraph;
using namespace ngraph::runtime::gpu;

CUDNNEmitter::CUDNNEmitter(GPUPrimitiveEmitter* emitter)
    : m_primitive_emitter(emitter)
{
}

size_t CUDNNEmitter::build_reduce_forward(GPURuntimeContext* ctx,
                                          const Shape& input_shape,
                                          const AxisSet& reduction_axes,
                                          const cudnnReduceTensorOp_t& reduce_op)
{
    std::function<cudnnTensorDescriptor_t(void)> get_input_desc;
    std::function<cudnnTensorDescriptor_t(void)> get_output_desc;
    if (input_shape.size() <= 4)
    {
        // construct input tensor descriptor rt impl.
        std::array<int, 4> dimensions;
        size_t pos = 0;
        for (size_t i = input_shape.size(); i < 4; i++)
        {
            dimensions[pos++] = 1;
        }
        for (size_t i = 0; i < input_shape.size(); i++)
        {
            dimensions[pos++] = static_cast<int>(input_shape[i]);
        }

        get_input_desc = [dimensions]() {
            cudnnTensorDescriptor_t desc;
            cudnnCreateTensorDescriptor(&desc);
            cudnnSetTensor4dDescriptor(desc,
                                       CUDNN_TENSOR_NCHW,
                                       CUDNN_DATA_FLOAT,
                                       dimensions[0],
                                       dimensions[1],
                                       dimensions[2],
                                       dimensions[3]);
            return desc;
        };

        // mark reduced axes of input tensor for output tensor descriptor
        for (auto const& idx_dim : reduction_axes)
        {
            dimensions[(4 - input_shape.size()) + idx_dim] = 1;
        }

        get_output_desc = [dimensions]() {
            cudnnTensorDescriptor_t desc;
            cudnnCreateTensorDescriptor(&desc);
            cudnnSetTensor4dDescriptor(desc,
                                       CUDNN_TENSOR_NCHW,
                                       CUDNN_DATA_FLOAT,
                                       dimensions[0],
                                       dimensions[1],
                                       dimensions[2],
                                       dimensions[3]);
            return desc;
        };
    }
    // descriptors for Nd tensors
    else
    {
        auto dimensions = cudnn_util::get_vector_int_from_size_t(input_shape);
        get_input_desc = [dimensions]() {
            cudnnTensorDescriptor_t desc;
            cudnnCreateTensorDescriptor(&desc);
            cudnnSetTensorNdDescriptor(desc,
                                       CUDNN_DATA_FLOAT,
                                       static_cast<int>(dimensions.size()),
                                       dimensions.data(),
                                       cudnn_util::compute_strides(dimensions).data());
            return desc;
        };

        // mark reduced axes of input tensor for output tensor descriptor
        for (auto const& idx_dim : reduction_axes)
        {
            dimensions[idx_dim] = 1;
        }

        get_output_desc = [dimensions]() {
            cudnnTensorDescriptor_t desc;
            cudnnCreateTensorDescriptor(&desc);
            cudnnSetTensorNdDescriptor(desc,
                                       CUDNN_DATA_FLOAT,
                                       static_cast<int>(dimensions.size()),
                                       dimensions.data(),
                                       cudnn_util::compute_strides(dimensions).data());
            return desc;
        };
    }
    // emit sum reduce operation
    auto* reduce = new gpu::primitive{
        [ctx, reduce_op, get_input_desc, get_output_desc](void** inputs, void** outputs) {
            auto input_desc = get_input_desc();
            auto output_desc = get_output_desc();
            cudnnReduceTensorDescriptor_t reduceTensorDesc;
            cudnnCreateReduceTensorDescriptor(&reduceTensorDesc);
            cudnnSetReduceTensorDescriptor(reduceTensorDesc,
                                           reduce_op,
                                           CUDNN_DATA_FLOAT,
                                           CUDNN_NOT_PROPAGATE_NAN,
                                           CUDNN_REDUCE_TENSOR_NO_INDICES,
                                           CUDNN_32BIT_INDICES);
            size_t workspace_size = 0;
            cudnnGetReductionWorkspaceSize(
                *ctx->cudnn_handle, reduceTensorDesc, input_desc, output_desc, &workspace_size);
            auto workspace_ptr = create_gpu_buffer(workspace_size);
            float alpha = 1.0, beta = 0.0;
            cudnnReduceTensor(*ctx->cudnn_handle,
                              reduceTensorDesc,
                              nullptr,
                              0,
                              workspace_ptr,
                              workspace_size,
                              &alpha,
                              input_desc,
                              inputs[0],
                              &beta,
                              output_desc,
                              outputs[0]);
            free_gpu_buffer(workspace_ptr);
        }};

    return this->m_primitive_emitter->insert(reduce);
}

std::vector<int> cudnn_util::compute_strides(const Shape& shape)
{
    return cudnn_util::get_vector_int_from_size_t(ngraph::row_major_strides(shape));
}

std::vector<int> cudnn_util::compute_strides(const std::vector<int>& shape)
{
    std::vector<int> strides(shape.size(), 1);
    std::copy(shape.begin() + 1, shape.end(), strides.begin());
    for (int64_t i = shape.size() - 2; i >= 0; i--)
    {
        strides[i] *= strides[i + 1];
    }
    return strides;
}

std::vector<int> cudnn_util::get_vector_int_from_size_t(const std::vector<size_t>& vec)
{
    std::vector<int> low_vec(vec.size(), 1);
    for (int i = 0; i < vec.size(); i++)
    {
        low_vec[i] = static_cast<int>(vec[i]);
    }
    return low_vec;
}
