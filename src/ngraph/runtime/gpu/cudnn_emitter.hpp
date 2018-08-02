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
#include "ngraph/runtime/gpu/cudnn_host_parameters.hpp"
#include "ngraph/runtime/gpu/gpu_runtime_context.hpp"
#include "ngraph/shape.hpp"

#include "ngraph/op/convolution.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/min.hpp"

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
                size_t build_primitive(const op::Convolution* node);
                size_t build_primitive(const op::ConvolutionBackpropData* node);
                size_t build_primitive(const op::ConvolutionBackpropFilters* node);
                size_t build_primitive(const op::MaxPool* node);
                size_t build_primitive(const op::Max* node);
                size_t build_primitive(const op::Min* node);

            public:
                enum class Prop
                {
                    Inference = 0,
                    Forward,
                    Backward
                };

                size_t build_convolution(const std::string& dtype,
                                         const Shape& input_tensor_shape,
                                         const Shape& input_filter_shape,
                                         const Shape& output_tensor_shape,
                                         const Strides& window_movement_strides,
                                         const Strides& window_dilation_strides,
                                         const Shape& padding_below);

                size_t build_convolution_backward_data(const std::string& dtype,
                                                       const Shape& input_filter_shape,
                                                       const Shape& input_tensor_shape,
                                                       const Shape& output_tensor_shape,
                                                       const Strides& window_movement_strides,
                                                       const Strides& window_dilation_strides,
                                                       const Shape& padding_below);

                size_t build_convolution_backward_filter(const std::string& dtype,
                                                         const Shape& input_tensor_shape_0,
                                                         const Shape& input_tensor_shape_1,
                                                         const Shape& output_filter_shape,
                                                         const Strides& window_movement_strides,
                                                         const Strides& window_dilation_strides,
                                                         const Shape& padding_below);

                size_t build_reduce_forward(const cudnnReduceTensorOp_t& reduce_op,
                                            const std::string& dtype,
                                            const Shape& input_shape,
                                            const AxisSet& reduction_axes);

                size_t build_tensor_op(const cudnnOpTensorOp_t& tensor_op,
                                       const std::string& dtype,
                                       const Shape& input_shape,
                                       const double alpha0,
                                       const double alpha1,
                                       const double beta);

                size_t build_pooling(const cudnnPoolingMode_t& pool_op,
                                     const std::string& dtype,
                                     const Prop& direction,
                                     const ngraph::Shape& input_shape,
                                     const ngraph::Shape& output_shape,
                                     const ngraph::Strides& window_strides,
                                     const ngraph::Shape& window_shape,
                                     const ngraph::Shape& padding_below,
                                     const ngraph::Shape& padding_above);

                size_t build_batchnorm(const cudnnBatchNormMode_t& bn_op,
                                       const std::string& dtype,
                                       const Prop& direction,
                                       const Shape& tensor_shape,
                                       const Shape& param_shape,
                                       double epsilon);

                size_t build_softmax(const cudnnSoftmaxAlgorithm_t& algorithm,
                                     const cudnnSoftmaxMode_t& mode,
                                     const std::string& dtype,
                                     const Prop& direction,
                                     const Shape& tensor_shape);

                void debug_sync();
                void sync();

            private:
                CUDNNEmitter(GPUPrimitiveEmitter* emitter, GPURuntimeContext* ctx);

                void* get_data_by_type(cudnnDataType_t data_type, double value);

                cudnnDataType_t get_cudnn_datatype(std::string dtype);

                cudnnTensorDescriptor_t&
                    tensor_descriptor_from_shape(const Shape& shape,
                                                 const cudnnDataType_t data_type,
                                                 const cudnnTensorFormat_t tensor_format);
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

                CUDNNDescriptors m_descriptors;
                CUDNNHostParameters m_host_parameters;

                GPUPrimitiveEmitter* m_primitive_emitter;
                GPURuntimeContext* m_ctx;
            };
        }
    }
}
