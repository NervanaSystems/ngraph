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
#include <cudnn_v7.h>

#include "ngraph/axis_set.hpp"
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
                cudnnTensorDescriptor_t tensor_descriptor_from_shape(const Shape& shape);
            }
            class GPUPrimitiveEmitter;

            class CUDNNEmitter
            {
                friend class GPUPrimitiveEmitter;

            public:
                enum class Prop
                {
                    Forward,
                    Backward
                };

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

            private:
                CUDNNEmitter(GPUPrimitiveEmitter* emitter);
                GPUPrimitiveEmitter* m_primitive_emitter;
            };
        }
    }
}
