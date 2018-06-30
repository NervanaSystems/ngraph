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

#include <array>
#include "ngraph/codegen/code_writer.hpp"
#include "ngraph/runtime/gpu/gpu_cuda_kernel_ops.hpp"
#include "ngraph/runtime/gpu/gpu_shape.hpp"
#include "ngraph/strides.hpp"

namespace ngraph
{
    class GPUShape;

    namespace runtime
    {
        namespace gpu
        {
            struct GPURuntimeContext;
            class GPUPrimitiveEmitter;

            class CUDAEmitter
            {
                friend class GPUPrimitiveEmitter;

            public:
                size_t build_pad(const GPURuntimeContext* ctx,
                                 const std::array<std::string, 2>& dtypes,
                                 GPUShape input_shape,
                                 GPUShape output_shape,
                                 GPUShape pad_below,
                                 GPUShape pad_above,
                                 GPUShape pad_interior,
                                 const std::string& pad_value = "");

                size_t build_pad_dynamic(const runtime::gpu::GPURuntimeContext* ctx,
                                         const std::array<std::string, 2>& dtypes,
                                         GPUShape input_shape,
                                         GPUShape output_shape,
                                         GPUShape padding_below,
                                         GPUShape padding_interior);

                size_t build_1d_max_pool(const GPURuntimeContext* ctx,
                                         const std::array<std::string, 2>& dtypes,
                                         GPUShape input_shape,
                                         GPUShape output_shape,
                                         size_t window_width,
                                         size_t window_stride);

                size_t build_avg_pool(const GPURuntimeContext* ctx,
                                      const std::array<std::string, 2>& dtypes,
                                      GPUShape input_shape,
                                      GPUShape output_shape,
                                      GPUShape window_shape,
                                      GPUShape window_stride,
                                      GPUShape padding_below,
                                      bool include_pad = false);

                size_t build_reduce_window(const GPURuntimeContext* ctx,
                                           const OpName op_name,
                                           const std::vector<std::string>& dtypes,
                                           GPUShape input_shape,
                                           GPUShape output_shape,
                                           GPUShape reduce_window_shape,
                                           GPUShape reduce_window_strides);

                size_t build_reverse_sequence(const runtime::gpu::GPURuntimeContext* ctx,
                                              const std::array<std::string, 3>& dtypes,
                                              GPUShape input_shape0,
                                              GPUShape input_shape1,
                                              GPUShape output_shape,
                                              size_t batch_axis,
                                              size_t sequence_axis);

                template <typename T>
                size_t build_elementwise(const GPURuntimeContext* ctx,
                                         const std::vector<std::string>& dtypes,
                                         GPUShape tensor_shape)
                {
                    return build_elementwise_n_to_1(
                        ctx, dtypes, tensor_shape, CudaOpMap<T>::op, CudaOpMap<T>::math_kernel);
                }

                template <typename ELEMENTWISE_OP_TYPE, typename REDUCE_OP_TYPE = ngraph::op::Nop>
                size_t build_elementwise_collective(const GPURuntimeContext* ctx,
                                                    const std::vector<std::string>& dtypes,
                                                    GPUShape tensor_shape,
                                                    const std::set<size_t>& reduced_tensors = {},
                                                    const std::set<size_t>& axes = {},
                                                    bool save_elementwise = false)
                {
                    return build_fused_ew_to_collective(ctx,
                                                        dtypes,
                                                        tensor_shape,
                                                        reduced_tensors,
                                                        axes,
                                                        CudaOpMap<ELEMENTWISE_OP_TYPE>::op,
                                                        CudaOpMap<ELEMENTWISE_OP_TYPE>::math_kernel,
                                                        CudaOpMap<REDUCE_OP_TYPE>::atomic,
                                                        save_elementwise);
                }

                size_t build_replace_slice(const GPURuntimeContext* ctx,
                                           const std::array<std::string, 3>& dtypes,
                                           GPUShape tensor_shape,
                                           GPUShape source_shape,
                                           GPUShape lower_bounds,
                                           GPUShape upper_bounds,
                                           GPUShape slice_stride);

                size_t build_broadcast(const GPURuntimeContext* ctx,
                                       const std::array<std::string, 2>& dtypes,
                                       GPUShape result_shape,
                                       const std::set<size_t>& bcast_axes);

                size_t build_reshape(const GPURuntimeContext* ctx,
                                     const std::array<std::string, 2>& dtypes,
                                     GPUShape input_shape,
                                     GPUShape input_order);

                size_t build_convolution(const GPURuntimeContext* ctx,
                                         const std::array<std::string, 3>& dtypes,
                                         GPUShape input_shape,
                                         GPUShape input_pad_below,
                                         GPUShape input_dilation,
                                         GPUShape filter_shape,
                                         GPUShape filter_stride,
                                         GPUShape filter_dilation,
                                         GPUShape output_shape);

            private:
                CUDAEmitter(GPUPrimitiveEmitter* emitter);
                void print_tensor_from_gpu(codegen::CodeWriter& writer,
                                           const std::string& tensor_name,
                                           GPUShape shape);
                std::string include_helpers();
                size_t build_elementwise_n_to_1(const GPURuntimeContext* ctx,
                                                const std::vector<std::string>& dtypes,
                                                GPUShape tensor_shape,
                                                const char* op,
                                                const char* kernel);
                size_t build_fused_ew_to_collective(const GPURuntimeContext* ctx,
                                                    const std::vector<std::string>& dtypes,
                                                    GPUShape tensor_shape,
                                                    const std::set<size_t>& reduced_tensors,
                                                    const std::set<size_t>& axes,
                                                    const char* op,
                                                    const char* kernel,
                                                    const char* reduce_op,
                                                    bool save_elementwise);

                GPUPrimitiveEmitter* m_primitive_emitter;
            };
        }
    }
}
