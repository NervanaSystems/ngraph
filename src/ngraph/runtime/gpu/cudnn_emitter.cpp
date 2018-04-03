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

#include "ngraph/util.hpp"
#include "ngraph/runtime/gpu/cudnn_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_runtime_context.hpp"

using namespace ngraph;
using namespace ngraph::runtime::gpu;

void CUDNNEmitter::invoke(size_t primitive_index,
                          const std::vector<void*>& args,
                          const std::vector<void*>& result)
{
    m_cudnn_primitives[primitive_index](args, result);
}

size_t CUDNNEmitter::register_primitive(
    const std::function<void(std::vector<void*>, std::vector<void*>)>& f)
{
    // try emplace
    m_cudnn_primitives.push_back(f);
    return m_cudnn_primitives.size() - 1;
}

size_t CUDNNEmitter::build_reduce_forward(cudnnReduceTensorOp_t reduce_op,
                                          const GPURuntimeContext* ctx,
                                          const ngraph::Shape& input_shape,
                                          const ngraph::AxisSet& reduction_axes)
{
    std::function<cudnnTensorDescriptor_t(void)> get_input_desc;
    std::function<cudnnTensorDescriptor_t(void)> get_output_desc;
    if (input_shape.size() <= 4)
    {
        // construct input tensor descriptor rt impl.
        std::array<size_t, 4> dimensions;
        size_t pos = 0;
        for (size_t i = input_shape.size(); i < 4; i++)
        {
            dimensions[pos++] = 1;
        }
        for (size_t i = 0; i < input_shape.size(); i++)
        {
            dimensions[pos++] = input_shape[i];
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
        std::vector<int> dimensions(input_shape.size());
        for (auto i = 0u; i < input_shape.size(); i++)
        {
            dimensions[i] = static_cast<int>(input_shape[i]);
        }
        get_input_desc = [dimensions]() {
            float* x = new float();

            cudnnTensorDescriptor_t desc;
            cudnnCreateTensorDescriptor(&desc);
            cudnnSetTensorNdDescriptor(desc,
                                       CUDNN_DATA_FLOAT,
                                       dimensions.size(),
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
                                       dimensions.size(),
                                       dimensions.data(),
                                       cudnn_util::compute_strides(dimensions).data());
            return desc;
        };
    }
    // emit sum reduce operation
    auto reduce = [ctx, reduce_op, get_input_desc, get_output_desc](std::vector<void*> inputs,
                                                                    std::vector<void*> outputs) {
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
    };

    return this->register_primitive(reduce);
}

std::vector<int> cudnn_util::compute_strides(const std::vector<int>& dim)
{
    std::vector<int> strides(dim.size(), 1);
    std::copy(dim.begin() + 1, dim.end(), strides.begin());
    for (int64_t i = dim.size() - 2; i >= 0; i--)
    {
        strides[i] *= strides[i + 1];
    }

    return strides;
}

size_t CUDNNEmitter::build_pooling_forward(cudnnPoolingMode_t pool_op,
                                           const GPURuntimeContext* ctx,
                                           const ngraph::Shape& input_shape,
                                           const ngraph::Shape& output_shape,
                                           const ngraph::Strides& window_strides,
                                           const ngraph::Shape& window_shape,
                                           const ngraph::Shape& padding_below,
                                           const ngraph::Shape& padding_above)
{
    if (input_shape.size() != 4)
    {
        // cudnn impl. currently only supportind 2d pooling
        // 1d must be added via cuda kernel
        // 3d can be added via cudnn
        throw std::runtime_error("Unsupported tensor encountered "
                                 "in CUDNNEmitter::build_pooling_forward.");
    }

    auto get_tensor_desc = [](const Shape& dimensions) {
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

    auto pool = [=](std::vector<void*> inputs,
                   std::vector<void*> outputs)
        {
            auto input_desc = get_tensor_desc(input_shape);
            auto output_desc = get_tensor_desc(output_shape);
            cudnnPoolingDescriptor_t desc;
            cudnnCreatePoolingDescriptor(&desc);
            cudnnSetPooling2dDescriptor(desc,
                                        pool_op,
                                        CUDNN_NOT_PROPAGATE_NAN,
                                        window_shape[0],
                                        window_shape[1],
                                        0,
                                        0,
                                        window_strides[0],
                                        window_strides[1]);

            float alpha = 1.0, beta = 0.0;
            cudnnPoolingForward(
                *ctx->cudnn_handle,
                desc,
                &alpha,
                input_desc,
                inputs[0],
                &beta,
                output_desc,
                outputs[0]);

    };

    return this->register_primitive(pool);
}
