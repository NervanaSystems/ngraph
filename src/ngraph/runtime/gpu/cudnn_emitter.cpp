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
#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>

#include "ngraph/runtime/gpu/cudnn_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_primitive_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_runtime_context.hpp"
#include "ngraph/runtime/gpu/gpu_util.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;

cudnnTensorDescriptor_t runtime::gpu::cudnn_util::tensor_descriptor_from_shape(const Shape& shape)
{
    cudnnTensorDescriptor_t desc;
    cudnnCreateTensorDescriptor(&desc);

    if (shape.size() < 4)
    {
        std::array<int, 4> dimensions;
        size_t pos = 0;
        for (size_t i = shape.size(); i < 4; i++)
        {
            dimensions[pos++] = 1;
        }
        for (size_t i = 0; i < shape.size(); i++)
        {
            dimensions[pos++] = static_cast<int>(shape[i]);
        }
        cudnnSetTensor4dDescriptor(desc,
                                   CUDNN_TENSOR_NCHW,
                                   CUDNN_DATA_FLOAT,
                                   dimensions[0],
                                   dimensions[1],
                                   dimensions[2],
                                   dimensions[3]);
    }
    else if (shape.size() == 4)
    {
        cudnnSetTensor4dDescriptor(desc,
                                   CUDNN_TENSOR_NCHW,
                                   CUDNN_DATA_FLOAT,
                                   static_cast<int>(shape[0]),
                                   static_cast<int>(shape[1]),
                                   static_cast<int>(shape[2]),
                                   static_cast<int>(shape[3]));
    }
    else
    {
        std::vector<int> dimensions(shape.size());
        for (auto i = 0u; i < shape.size(); i++)
        {
            dimensions[i] = static_cast<int>(shape[i]);
        }
        cudnnSetTensorNdDescriptor(desc,
                                   CUDNN_DATA_FLOAT,
                                   static_cast<int>(dimensions.size()),
                                   dimensions.data(),
                                   runtime::gpu::cudnn_util::compute_strides(dimensions).data());
    }

    return desc;
}

std::vector<int> runtime::gpu::cudnn_util::compute_strides(const Shape& shape)
{
    return runtime::gpu::cudnn_util::get_vector_int_from_size_t(row_major_strides(shape));
}

std::vector<int> runtime::gpu::cudnn_util::compute_strides(const std::vector<int>& shape)
{
    std::vector<int> strides(shape.size(), 1);
    std::copy(shape.begin() + 1, shape.end(), strides.begin());
    for (int64_t i = shape.size() - 2; i >= 0; i--)
    {
        strides[i] *= strides[i + 1];
    }
    return strides;
}

std::vector<int>
    runtime::gpu::cudnn_util::get_vector_int_from_size_t(const std::vector<size_t>& vec)
{
    std::vector<int> low_vec(vec.size(), 1);
    for (int i = 0; i < vec.size(); i++)
    {
        low_vec[i] = static_cast<int>(vec[i]);
    }
    return low_vec;
}

runtime::gpu::CUDNNEmitter::CUDNNEmitter(GPUPrimitiveEmitter* emitter)
    : m_primitive_emitter(emitter)
{
}

size_t runtime::gpu::CUDNNEmitter::build_reduce_forward(const runtime::gpu::GPURuntimeContext* ctx,
                                                        const cudnnReduceTensorOp_t& reduce_op,
                                                        const Shape& input_shape,
                                                        const AxisSet& reduction_axes)
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
        auto dimensions = runtime::gpu::cudnn_util::get_vector_int_from_size_t(input_shape);
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

size_t runtime::gpu::CUDNNEmitter::build_pooling(const runtime::gpu::GPURuntimeContext* ctx,
                                                 const cudnnPoolingMode_t& pool_op,
                                                 const Prop& direction,
                                                 const Shape& input_shape,
                                                 const Shape& output_shape,
                                                 const Strides& window_strides,
                                                 const Shape& window_shape,
                                                 const Shape& padding_below,
                                                 const Shape& padding_above)
{
    // construct hash to determine if kernel needs to be emitted
    // or if it already exists in the primitive list
    std::stringstream ss;
    ss << "pool_op" << pool_op << "_dir" << static_cast<int>(direction) << "_i"
       << join(input_shape, "_") << "_o" << join(output_shape, "_") << "_ws"
       << join(window_shape, "_") << "_wst" << join(window_strides, "_") << "_pb"
       << join(padding_below, "_") << "_pb" << join(padding_above, "_");
    std::string hash = ss.str();

    // check if the requested kernel is already an inserted primitive
    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    cudnnPoolingDescriptor_t desc;
    auto input_desc = runtime::gpu::cudnn_util::tensor_descriptor_from_shape(input_shape);
    auto output_desc = runtime::gpu::cudnn_util::tensor_descriptor_from_shape(output_shape);
    if (input_shape.size() == 4)
    {
        cudnnCreatePoolingDescriptor(&desc);
        cudnnSetPooling2dDescriptor(desc,
                                    pool_op,
                                    CUDNN_NOT_PROPAGATE_NAN,
                                    static_cast<int>(window_shape[0]),
                                    static_cast<int>(window_shape[1]),
                                    static_cast<int>(padding_below[0]),
                                    static_cast<int>(padding_below[1]),
                                    static_cast<int>(window_strides[0]),
                                    static_cast<int>(window_strides[1]));
    }
    else if (input_shape.size() == 5)
    {
        std::vector<int> w_strides(window_strides.size());
        std::vector<int> w_shape(window_shape.size());
        std::vector<int> w_padding(padding_below.size());
        for (int i = 0; i < window_shape.size(); i++)
        {
            w_shape[i] = static_cast<int>(window_shape[i]);
            w_strides[i] = static_cast<int>(window_strides[i]);
            w_padding[i] = static_cast<int>(padding_below[i]);
        }
        cudnnCreatePoolingDescriptor(&desc);
        cudnnSetPoolingNdDescriptor(desc,
                                    pool_op,
                                    CUDNN_NOT_PROPAGATE_NAN,
                                    3,
                                    w_shape.data(),
                                    w_padding.data(),
                                    w_strides.data());
    }
    else
    {
        throw std::runtime_error("Pooling currently supports up to 3 spatial dimensions only.");
    }

    gpu::primitive* pool = nullptr;
    if (direction != Prop::Backward)
    {
        pool = new gpu::primitive{[=](void** inputs, void** outputs) {
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
    else
    {
        pool = new gpu::primitive{[=](void** inputs, void** outputs) {
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

    primitive_index = this->m_primitive_emitter->insert(pool);
    m_primitive_emitter->cache(hash, primitive_index);
    return primitive_index;
}

size_t runtime::gpu::CUDNNEmitter::build_batchnorm(const runtime::gpu::GPURuntimeContext* ctx,
                                                   const cudnnBatchNormMode_t& bn_op,
                                                   const Prop& direction,
                                                   const Shape& tensor_shape,
                                                   const Shape& param_shape,
                                                   double epsilon)
{
    std::stringstream ss;
    // hash params TBD
    std::string hash = ss.str();

    size_t primitive_index = m_primitive_emitter->lookup(hash);
    if (primitive_index != std::numeric_limits<size_t>::max())
    {
        return primitive_index;
    }

    if (epsilon < CUDNN_BN_MIN_EPSILON)
    {
        throw std::runtime_error("Batch Norm epsilon is less than CUDNN_BN_MIN_EPSILON");
    }

    cudnnTensorDescriptor_t derived_param_desc;
    cudnnCreateTensorDescriptor(&derived_param_desc);
    auto tensor_desc = runtime::gpu::cudnn_util::tensor_descriptor_from_shape(tensor_shape);
    cudnnDeriveBNTensorDescriptor(derived_param_desc, tensor_desc, bn_op);

    float alpha = 1.0, beta = 0.0;
    gpu::primitive* batchnorm;
    switch (direction)
    {
    case Prop::Inference: {
        batchnorm = new gpu::primitive{[=](void** inputs, void** outputs) {
                cudnnBatchNormalizationForwardInference(*ctx->cudnn_handle,
                                                        bn_op,
                                                        &alpha,
                                                        &beta,
                                                        tensor_desc,
                                                        inputs[2],  // tensor
                                                        tensor_desc,
                                                        outputs[0], // tensor
                                                        derived_param_desc,
                                                        inputs[0],  // gain
                                                        inputs[1],  // bias
                                                        inputs[3],  // mean
                                                        inputs[4],  // variance
                                                        epsilon);
            }};
        break;
    }
    case Prop::Forward: {
        // currently not using the cudnn calculation of mean/variance
        // so this factor needs to be set to 1.0;
        double exp_avg_factor = 1.0f;
        batchnorm = new gpu::primitive{[=](void** inputs, void** outputs) {
                cudnnBatchNormalizationForwardTraining(
                    *ctx->cudnn_handle,
                    bn_op,
                    &alpha,
                    &beta,
                    tensor_desc,
                    inputs[2],
                    tensor_desc,
                    outputs[0],
                    derived_param_desc,
                    inputs[0],
                    inputs[1],
                    exp_avg_factor,
                    outputs[1], // running mean
                    outputs[2], // running var
                    epsilon,
                    NULL,       // batch mean
                    NULL);      // batch var
            }};
        break;
    }
    case Prop::Backward: {
        batchnorm = new gpu::primitive{[=](void** inputs, void** outputs) {
                cudnnBatchNormalizationBackward(
                    *ctx->cudnn_handle,
                    bn_op,
                    &alpha,
                    &beta,
                    &alpha,
                    &beta,
                    tensor_desc,
                    inputs[2 /* input tensor x */],
                    tensor_desc,
                    inputs[5 /* dy */],
                    tensor_desc,
                    outputs[0/* dx */],
                    derived_param_desc,
                    inputs[0 /* gamma */],
                    outputs[1 /* dgamma */],
                    outputs[2 /* dbeta */],
                    epsilon,
                    inputs[3 /* mu batch mean*/],
                    inputs[4 /* 1/sig**2 batch inverse variance*/]);
            }};
        break;
    }
    default: {
        throw std::runtime_error("No default case for CUDNN BatchNorm");
    }
    }


    primitive_index = this->m_primitive_emitter->insert(batchnorm);
    m_primitive_emitter->cache(hash, primitive_index);
    return primitive_index;
}
