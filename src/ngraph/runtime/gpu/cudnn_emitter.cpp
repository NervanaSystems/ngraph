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
#include "ngraph/runtime/gpu/gpu_util.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace ngraph::runtime::gpu;

CUDNNEmitter::CUDNNEmitter(GPUPrimitiveEmitter* emitter)
    : m_primitive_emitter(emitter)
{
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
    gpu::primitive* fpool = nullptr;
    if (input_shape.size() == 4)
    {
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

        fpool = new gpu::primitive{[=](void** inputs, void** outputs) {
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
            cudnnPoolingForward(*ctx->cudnn_handle,
                                desc,
                                &alpha,
                                input_desc,
                                inputs[0],
                                &beta,
                                output_desc,
                                outputs[0]);
        }};
    }
    else if (input_shape.size() == 5)
    {
        if (window_shape.size() != 3 || window_strides.size() != 3 || padding_below.size() != 3)
        {
            throw std::runtime_error(
                "3d pooling requested but window properties are not 3 dimensional.");
        }
        auto get_tensor_desc = [](const Shape& shape) {
            std::vector<int> dimensions(shape.size());
            for (auto i = 0u; i < shape.size(); i++)
            {
                dimensions[i] = static_cast<int>(shape[i]);
            }
            cudnnTensorDescriptor_t desc;
            cudnnCreateTensorDescriptor(&desc);
            cudnnSetTensorNdDescriptor(desc,
                                       CUDNN_DATA_FLOAT,
                                       dimensions.size(),
                                       dimensions.data(),
                                       cudnn_util::compute_strides(dimensions).data());
            return desc;
        };

        std::vector<int> w_strides(window_strides.size());
        std::vector<int> w_shape(window_shape.size());
        std::vector<int> w_padding(padding_below.size());
        for (int i = 0; i < window_shape.size(); i++)
        {
            w_shape[i] = static_cast<int>(window_shape[i]);
            w_strides[i] = static_cast<int>(window_strides[i]);
            w_padding[i] = static_cast<int>(padding_below[i]);
        }

        fpool = new gpu::primitive{[=](void** inputs, void** outputs) {
            auto input_desc = get_tensor_desc(input_shape);
            auto output_desc = get_tensor_desc(output_shape);
            cudnnPoolingDescriptor_t desc;
            cudnnCreatePoolingDescriptor(&desc);
            cudnnSetPoolingNdDescriptor(desc,
                                        pool_op,
                                        CUDNN_NOT_PROPAGATE_NAN,
                                        3,
                                        w_shape.data(),
                                        w_padding.data(),
                                        w_strides.data());

            float alpha = 1.0, beta = 0.0;
            cudnnPoolingForward(*ctx->cudnn_handle,
                                desc,
                                &alpha,
                                input_desc,
                                inputs[0],
                                &beta,
                                output_desc,
                                outputs[0]);
        }};
    }

    return this->m_primitive_emitter->insert(fpool);
}

size_t CUDNNEmitter::build_pooling_backward(cudnnPoolingMode_t pool_op,
                                            const GPURuntimeContext* ctx,
                                            const ngraph::Shape& input_shape,
                                            const ngraph::Shape& output_shape,
                                            const ngraph::Strides& window_strides,
                                            const ngraph::Shape& window_shape,
                                            const ngraph::Shape& padding_below,
                                            const ngraph::Shape& padding_above)
{
    gpu::primitive* bpool = nullptr;
    if (input_shape.size() == 4)
    {
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

        bpool = new gpu::primitive{[=](void** inputs, void** outputs) {
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
            // cuDNN requires the output tensor of the maxpool fprop to be passed even though
            // it is not mathematically necessary. It appears, however, that it is not actually
            // used as the adjoints are passed in place and the correct result is achieved.
            cudnnPoolingBackward(*ctx->cudnn_handle,
                                 desc,
                                 &alpha,
                                 // output (wrt maxpool) tensor
                                 output_desc,
                                 inputs[1],
                                 // adjoint of output
                                 output_desc,
                                 inputs[1],
                                 // input (wrt maxpool) tensor
                                 input_desc,
                                 inputs[0],
                                 &beta,
                                 // adjoint of input
                                 input_desc,
                                 outputs[0]);
        }};
    }
    else if (input_shape.size() == 5)
    {
        if (window_shape.size() != 3 || window_strides.size() != 3 || padding_below.size() != 3)
        {
            throw std::runtime_error(
                "3d pooling requested but window properties are not 3 dimensional.");
        }
        auto get_tensor_desc = [](const Shape& shape) {
            std::vector<int> dimensions(shape.size());
            for (auto i = 0u; i < shape.size(); i++)
            {
                dimensions[i] = static_cast<int>(shape[i]);
            }
            cudnnTensorDescriptor_t desc;
            cudnnCreateTensorDescriptor(&desc);
            cudnnSetTensorNdDescriptor(desc,
                                       CUDNN_DATA_FLOAT,
                                       dimensions.size(),
                                       dimensions.data(),
                                       cudnn_util::compute_strides(dimensions).data());
            return desc;
        };

        std::vector<int> w_strides(window_strides.size());
        std::vector<int> w_shape(window_shape.size());
        std::vector<int> w_padding(padding_below.size());
        for (int i = 0; i < window_shape.size(); i++)
        {
            w_shape[i] = static_cast<int>(window_shape[i]);
            w_strides[i] = static_cast<int>(window_strides[i]);
            w_padding[i] = static_cast<int>(padding_below[i]);
        }

        bpool = new gpu::primitive{[=](void** inputs, void** outputs) {
            auto input_desc = get_tensor_desc(input_shape);
            auto output_desc = get_tensor_desc(output_shape);
            cudnnPoolingDescriptor_t desc;
            cudnnCreatePoolingDescriptor(&desc);
            cudnnSetPoolingNdDescriptor(desc,
                                        pool_op,
                                        CUDNN_NOT_PROPAGATE_NAN,
                                        3,
                                        w_shape.data(),
                                        w_padding.data(),
                                        w_strides.data());

            float alpha = 1.0, beta = 0.0;
            // cuDNN requires the output tensor of the maxpool fprop to be passed even though
            // it is not mathematically necessary. It appears, however, that it is not actually
            // used as the adjoints are passed in place and the correct result is achieved.
            cudnnPoolingBackward(*ctx->cudnn_handle,
                                 desc,
                                 &alpha,
                                 // output (wrt maxpool) tensor
                                 output_desc,
                                 inputs[1],
                                 // adjoint of output
                                 output_desc,
                                 inputs[1],
                                 // input (wrt maxpool) tensor
                                 input_desc,
                                 inputs[0],
                                 &beta,
                                 // adjoint of input
                                 input_desc,
                                 outputs[0]);
        }};
    }

    return this->m_primitive_emitter->insert(bpool);
}
