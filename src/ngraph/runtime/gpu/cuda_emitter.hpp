//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <array>
#include "ngraph/codegen/code_writer.hpp"
#include "ngraph/runtime/gpu/gpu_cuda_kernel_ops.hpp"
#include "ngraph/runtime/gpu/nvdiff.hpp"
#include "ngraph/runtime/gpu/nvshape.hpp"
#include "ngraph/strides.hpp"

#include "ngraph/op/convolution.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/softmax.hpp"

namespace ngraph
{
    class NVShape;

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
                size_t build_primitive(const op::Softmax* node);
                size_t build_primitive(const op::Convolution* node);
                size_t build_primitive(const op::MaxPool* node);
                size_t build_primitive(const op::ReplaceSlice* node, bool in_place_op);

            public:
                size_t build_topk(const std::vector<element::Type>& dtypes,
                                  const NVShape& input_shape,
                                  const size_t topk_axis,
                                  size_t topk_k,
                                  const element::Type index_elem_type,
                                  bool compute_max);

                size_t build_pad(const std::vector<std::string>& dtypes,
                                 NVShape input_shape,
                                 NVShape output_shape,
                                 NVShape padding_below,
                                 NVShape padding_interior);

                size_t build_pad_fill(const std::vector<std::string>& dtypes,
                                      NVShape input_shape,
                                      NVShape output_shape,
                                      NVShape padding_below,
                                      NVShape padding_interior);

                size_t build_1d_max_pool(const std::array<std::string, 2>& dtypes,
                                         NVShape input_shape,
                                         NVShape output_shape,
                                         size_t window_width,
                                         size_t window_stride);

                size_t build_avg_pool(const std::array<std::string, 2>& dtypes,
                                      NVShape input_shape,
                                      NVShape output_shape,
                                      NVShape window_shape,
                                      NVShape window_stride,
                                      NVShape padding_below,
                                      bool include_pad = false);

                size_t build_slice(const std::array<std::string, 2>& dtypes,
                                   NVShape input_shape,
                                   NVShape lower_bounds,
                                   NVShape slice_strides,
                                   NVShape output_shape);

                size_t build_reduce_window(const OpName op_name,
                                           const std::vector<std::string>& dtypes,
                                           NVShape input_shape,
                                           NVShape output_shape,
                                           NVShape reduce_window_shape,
                                           NVShape reduce_window_strides);

                size_t build_reverse_sequence(const std::array<std::string, 3>& dtypes,
                                              NVShape input_shape0,
                                              NVShape input_shape1,
                                              NVShape output_shape,
                                              size_t batch_axis,
                                              size_t sequence_axis);

                size_t build_onehot(const std::array<std::string, 2>& dtypes,
                                    NVShape input_shape,
                                    NVShape output_shape,
                                    size_t one_hot_axis,
                                    size_t output_datatype_size);

                size_t build_reverse(const std::array<std::string, 2>& dtypes,
                                     NVShape input_shape,
                                     std::vector<uint32_t> reverse_axes);

                template <typename T>
                size_t build_elementwise(const std::vector<std::string>& dtypes,
                                         NVShape tensor_shape)
                {
                    return build_elementwise_n_to_1(
                        dtypes, tensor_shape, CudaOpMap<T>::op, CudaOpMap<T>::math_kernel);
                }

                size_t build_cudnn_bn_inv_var(const std::vector<std::string>& dtypes,
                                              NVShape tensor_shape,
                                              const double& eps);

                template <typename T>
                size_t build_reduce(const std::vector<std::string>& dtypes,
                                    const size_t data_bytes,
                                    NVShape input_shape,
                                    NVShape reduce_axis)
                {
                    return build_reduce(dtypes,
                                        data_bytes,
                                        input_shape,
                                        reduce_axis,
                                        CudaOpMap<T>::op,
                                        CudaOpMap<T>::math_kernel);
                }

                template <typename ELEMENTWISE_OP_TYPE, typename REDUCE_OP_TYPE = ngraph::op::Nop>
                size_t build_elementwise_collective(const std::vector<std::string>& dtypes,
                                                    NVShape tensor_shape,
                                                    const std::set<size_t>& reduced_tensors = {},
                                                    const std::set<size_t>& axes = {},
                                                    bool save_elementwise = false)
                {
                    return build_fused_ew_to_collective(dtypes,
                                                        tensor_shape,
                                                        reduced_tensors,
                                                        axes,
                                                        CudaOpMap<ELEMENTWISE_OP_TYPE>::op,
                                                        CudaOpMap<ELEMENTWISE_OP_TYPE>::math_kernel,
                                                        CudaOpMap<REDUCE_OP_TYPE>::atomic,
                                                        save_elementwise);
                }

                size_t build_broadcast(const std::array<std::string, 2>& dtypes,
                                       NVShape result_shape,
                                       const std::set<size_t>& bcast_axes);

                size_t build_reshape(const std::array<std::string, 2>& dtypes,
                                     NVShape input_shape,
                                     NVShape input_order);

                size_t build_reshape_2d(const std::array<std::string, 2>& dtypes,
                                        NVShape input_shape,
                                        NVShape input_order);

                size_t build_reshape_3d(const std::array<std::string, 2>& dtypes,
                                        NVShape input_shape,
                                        NVShape input_order);

                size_t build_convolution(const std::array<std::string, 3>& dtypes,
                                         NVShape input_shape,
                                         NVShape filter_shape,
                                         NVShape output_shape,
                                         NVShape filter_stride,
                                         NVShape filter_dilation,
                                         NVShape input_dilation,
                                         NVDiff input_pad_below);

                size_t build_concat(const std::string& dtype,
                                    std::vector<NVShape> input_shapes,
                                    size_t concat_axis,
                                    NVShape output_shape);

                size_t build_softmax_divide(const std::vector<std::string>& dtypes,
                                            NVShape input_shape,
                                            NVShape reduce_shape,
                                            std::vector<size_t> axes_flag);

                void debug_sync();
                void sync();

            private:
                CUDAEmitter(GPUPrimitiveEmitter* emitter, GPURuntimeContext* ctx);
                uint32_t align_to_block_size(uint32_t threads, uint32_t block_size);
                void print_tensor_from_gpu(codegen::CodeWriter& writer,
                                           const std::string& tensor_name,
                                           NVShape shape);
                std::string include_helpers();
                size_t build_elementwise_n_to_1(const std::vector<std::string>& dtypes,
                                                NVShape tensor_shape,
                                                const char* op,
                                                const char* kernel);
                size_t build_fused_ew_to_collective(const std::vector<std::string>& dtypes,
                                                    NVShape tensor_shape,
                                                    const std::set<size_t>& reduced_tensors,
                                                    const std::set<size_t>& axes,
                                                    const char* op,
                                                    const char* kernel,
                                                    const char* reduce_op,
                                                    bool save_elementwise);
                size_t build_reduce(const std::vector<std::string>& dtypes,
                                    const size_t data_bytes,
                                    NVShape input_shape,
                                    NVShape reduce_axis,
                                    const char* op,
                                    const char* kernel);
                size_t build_reduce_to_nd(const std::vector<std::string>& dtypes,
                                          NVShape input_shape,
                                          NVShape reduce_axis,
                                          const char* op,
                                          const char* kernel);
                size_t build_reduce_to_scalar(const std::vector<std::string>& dtypes,
                                              const size_t data_bytes,
                                              NVShape input_shape,
                                              const char* op,
                                              const char* kernel);

                //This is the preprocess for reduce to scalar if the data size is large than a number.
                //The number can be tuned based on hardware.
                //This cuda kernel will accumulate reduction to a certain number of bins depends on hardware.
                size_t build_reduce_to_scalar_acc(const std::vector<std::string>& dtypes,
                                                  NVShape input_shape,
                                                  NVShape output_shape,
                                                  uint32_t block_size_x,
                                                  const char* op,
                                                  const char* kernel);
                GPUPrimitiveEmitter* m_primitive_emitter;
                GPURuntimeContext* m_ctx;
            };
        }
    }
}
