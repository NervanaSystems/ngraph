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

#pragma once

#include <functional>
#include <vector>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include "ngraph/axis_set.hpp"
#include "ngraph/runtime/gpu/cudnn_descriptors.hpp"
#include "ngraph/runtime/gpu/gpu_runtime_context.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            namespace cudnn_util
            {
                std::vector<int> compute_strides(const Shape&);
                std::vector<int> compute_strides(const std::vector<int>&);
                std::vector<int> get_vector_int_from_size_t(const std::vector<size_t>&);
            }
            class GPUPrimitiveEmitter;

            class CUDNNEmitter
            {
                friend class GPUPrimitiveEmitter;

            public:
                enum class Prop
                {
                    Inference,
                    Forward,
                    Backward,
                    BackwardFilter
                };

                size_t build_convolution(const runtime::gpu::GPURuntimeContext* ctx,
                                         const cudnnDataType_t data_type,
                                         const Prop& direction,
                                         const Shape& input_shape,
                                         const Shape& filter_shape,
                                         const Shape& output_shape,
                                         const Strides& window_movement_strides,
                                         const Strides& window_dilation_strides,
                                         const Shape& padding_below);

                size_t build_reduce_forward(const GPURuntimeContext* ctx,
                                            const cudnnReduceTensorOp_t& reduce_op,
                                            const Shape& input_shape,
                                            const AxisSet& reduction_axes);

                size_t build_pooling(const GPURuntimeContext* ctx,
                                     const cudnnPoolingMode_t& pool_op,
                                     const Prop& direction,
                                     const ngraph::Shape& input_shape,
                                     const ngraph::Shape& output_shape,
                                     const ngraph::Strides& window_strides,
                                     const ngraph::Shape& window_shape,
                                     const ngraph::Shape& padding_below,
                                     const ngraph::Shape& padding_above);

                size_t build_batchnorm(const runtime::gpu::GPURuntimeContext* ctx,
                                       const cudnnBatchNormMode_t& bn_op,
                                       const Prop& direction,
                                       const Shape& tensor_shape,
                                       const Shape& param_shape,
                                       double epsilon);

                size_t build_softmax(const runtime::gpu::GPURuntimeContext* ctx,
                                     const cudnnSoftmaxAlgorithm_t& algorithm,
                                     const cudnnSoftmaxMode_t& mode,
                                     const Prop& direction,
                                     const Shape& tensor_shape);

                cudnnTensorDescriptor_t& tensor_descriptor_from_shape(const Shape& shape);
                cudnnFilterDescriptor_t&
                    get_cudnn_filter_descriptor(const Shape& shape,
                                                const cudnnDataType_t data_type,
                                                const cudnnTensorFormat_t tensor_format);
                cudnnConvolutionDescriptor_t&
                    get_cudnn_convolution_descriptor(const Shape& padding,
                                                     const Strides& window_movement_strides,
                                                     const Strides& window_dilation_strides,
                                                     cudnnConvolutionMode_t mode,
                                                     cudnnDataType_t data_type);

            private:
                CUDNNEmitter(GPUPrimitiveEmitter* emitter);

                CUDNNDescriptors m_descriptors;
                GPUPrimitiveEmitter* m_primitive_emitter;
            };
        }
    }
}
