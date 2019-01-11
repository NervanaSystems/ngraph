//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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
#include "ngraph/runtime/nvidiagpu/nvdiff.hpp"
#include "ngraph/runtime/nvidiagpu/nvidiagpu_cuda_kernel_ops.hpp"
#include "ngraph/runtime/nvidiagpu/nvidiagpu_host_parameters.hpp"
#include "ngraph/runtime/nvidiagpu/nvshape.hpp"
#include "ngraph/strides.hpp"

#include "ngraph/op/convolution.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/softmax.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace nvidiagpu
        {
            class Shape;
            struct RuntimeContext;
            class PrimitiveEmitter;

            class CUDAEmitter
            {
                friend class PrimitiveEmitter;

            public:
                size_t build_primitive(const op::Convolution* node);
                size_t build_primitive(const op::MaxPool* node);
                size_t build_primitive(const op::ReplaceSlice* node, bool in_place_op);

            public:
                size_t build_memset(const std::string& dtype, uint32_t tensor_size);

                size_t build_topk(const std::vector<element::Type>& dtypes,
                                  const runtime::nvidiagpu::Shape& input_shape,
                                  const size_t topk_axis,
                                  size_t topk_k,
                                  const element::Type index_elem_type,
                                  bool compute_max);

                size_t build_pad(const std::vector<std::string>& dtypes,
                                 runtime::nvidiagpu::Shape input_shape,
                                 runtime::nvidiagpu::Shape output_shape,
                                 runtime::nvidiagpu::Shape padding_below,
                                 runtime::nvidiagpu::Shape padding_interior);

                size_t build_pad_fill(const std::vector<std::string>& dtypes,
                                      runtime::nvidiagpu::Shape input_shape,
                                      runtime::nvidiagpu::Shape output_shape,
                                      runtime::nvidiagpu::Shape padding_below,
                                      runtime::nvidiagpu::Shape padding_interior);

                size_t build_1d_max_pool(const std::array<std::string, 2>& dtypes,
                                         runtime::nvidiagpu::Shape input_shape,
                                         runtime::nvidiagpu::Shape output_shape,
                                         size_t window_width,
                                         size_t window_stride);

                size_t build_avg_pool(const std::array<std::string, 2>& dtypes,
                                      runtime::nvidiagpu::Shape input_shape,
                                      runtime::nvidiagpu::Shape output_shape,
                                      runtime::nvidiagpu::Shape window_shape,
                                      runtime::nvidiagpu::Shape window_stride,
                                      runtime::nvidiagpu::Shape padding_below,
                                      bool include_pad = false);

                size_t build_slice(const std::array<std::string, 2>& dtypes,
                                   runtime::nvidiagpu::Shape input_shape,
                                   runtime::nvidiagpu::Shape lower_bounds,
                                   runtime::nvidiagpu::Shape slice_strides,
                                   runtime::nvidiagpu::Shape output_shape);

                size_t build_reduce_window(const OpName op_name,
                                           const std::vector<std::string>& dtypes,
                                           runtime::nvidiagpu::Shape input_shape,
                                           runtime::nvidiagpu::Shape output_shape,
                                           runtime::nvidiagpu::Shape reduce_window_shape,
                                           runtime::nvidiagpu::Shape reduce_window_strides);

                size_t build_reverse_sequence(const std::array<std::string, 3>& dtypes,
                                              runtime::nvidiagpu::Shape input_shape0,
                                              runtime::nvidiagpu::Shape input_shape1,
                                              runtime::nvidiagpu::Shape output_shape,
                                              size_t batch_axis,
                                              size_t sequence_axis);

                size_t build_onehot(const std::array<std::string, 2>& dtypes,
                                    runtime::nvidiagpu::Shape input_shape,
                                    runtime::nvidiagpu::Shape output_shape,
                                    size_t one_hot_axis,
                                    size_t output_datatype_size);

                size_t build_reverse(const std::array<std::string, 2>& dtypes,
                                     runtime::nvidiagpu::Shape input_shape,
                                     std::vector<uint32_t> reverse_axes);

                template <typename T>
                size_t build_elementwise(const std::vector<std::string>& dtypes,
                                         runtime::nvidiagpu::Shape tensor_shape)
                {
                    return build_elementwise_n_to_1(
                        dtypes, tensor_shape, CudaOpMap<T>::op, CudaOpMap<T>::math_kernel);
                }

                size_t build_cudnn_bn_inv_var(const std::vector<std::string>& dtypes,
                                              runtime::nvidiagpu::Shape tensor_shape,
                                              const double& eps);

                template <typename T>
                size_t build_reduce(const std::vector<element::Type>& dtypes,
                                    runtime::nvidiagpu::Shape input_shape,
                                    runtime::nvidiagpu::Shape output_shape,
                                    runtime::nvidiagpu::Shape reduce_axis,
                                    const bool with_init_value = false)
                {
                    return build_reduce(dtypes,
                                        input_shape,
                                        output_shape,
                                        reduce_axis,
                                        CudaOpMap<T>::op,
                                        CudaOpMap<T>::math_kernel,
                                        with_init_value);
                }

                template <typename ELEMENTWISE_OP_TYPE, typename REDUCE_OP_TYPE = ngraph::op::Nop>
                size_t build_elementwise_collective(const std::vector<std::string>& dtypes,
                                                    runtime::nvidiagpu::Shape tensor_shape,
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
                                       runtime::nvidiagpu::Shape result_shape,
                                       const std::set<size_t>& bcast_axes);

                size_t build_reshape(const std::array<std::string, 2>& dtypes,
                                     runtime::nvidiagpu::Shape input_shape,
                                     runtime::nvidiagpu::Shape input_order);

                size_t build_reshape_2d(const std::array<std::string, 2>& dtypes,
                                        runtime::nvidiagpu::Shape input_shape,
                                        runtime::nvidiagpu::Shape input_order);

                size_t build_reshape_3d(const std::array<std::string, 2>& dtypes,
                                        runtime::nvidiagpu::Shape input_shape,
                                        runtime::nvidiagpu::Shape input_order);

                size_t build_convolution(const std::array<std::string, 3>& dtypes,
                                         runtime::nvidiagpu::Shape input_shape,
                                         runtime::nvidiagpu::Shape filter_shape,
                                         runtime::nvidiagpu::Shape output_shape,
                                         runtime::nvidiagpu::Shape filter_stride,
                                         runtime::nvidiagpu::Shape filter_dilation,
                                         runtime::nvidiagpu::Shape input_dilation,
                                         NVDiff input_pad_below);

                size_t build_concat(const std::string& dtype,
                                    std::vector<runtime::nvidiagpu::Shape> input_shapes,
                                    size_t concat_axis,
                                    runtime::nvidiagpu::Shape output_shape);

                size_t build_softmax(const std::vector<element::Type>& dtypes,
                                     runtime::nvidiagpu::Shape input_shape,
                                     runtime::nvidiagpu::Shape reduce_axis);

                void debug_sync();
                void sync();

            private:
                CUDAEmitter(PrimitiveEmitter* emitter,
                            RuntimeContext* ctx,
                            std::shared_ptr<HostParameters> params);
                uint32_t align_to_block_size(uint32_t threads, uint32_t block_size);
                void print_tensor_from_nvidiagpu(codegen::CodeWriter& writer,
                                             const std::string& tensor_name,
                                             runtime::nvidiagpu::Shape shape);
                std::string include_helpers();
                size_t build_elementwise_n_to_1(const std::vector<std::string>& dtypes,
                                                runtime::nvidiagpu::Shape tensor_shape,
                                                const char* op,
                                                const char* kernel);
                size_t build_fused_ew_to_collective(const std::vector<std::string>& dtypes,
                                                    runtime::nvidiagpu::Shape tensor_shape,
                                                    const std::set<size_t>& reduced_tensors,
                                                    const std::set<size_t>& axes,
                                                    const char* op,
                                                    const char* kernel,
                                                    const char* reduce_op,
                                                    bool save_elementwise);

                size_t build_reduce(const std::vector<element::Type>& dtypes,
                                    const runtime::nvidiagpu::Shape& input_shape,
                                    const runtime::nvidiagpu::Shape& output_shape,
                                    const runtime::nvidiagpu::Shape& reduce_axis,
                                    const char* op,
                                    const char* kernel,
                                    const bool with_init_value);
                size_t build_reduce_to_nd(const std::vector<element::Type>& dtypes,
                                          runtime::nvidiagpu::Shape input_shape,
                                          runtime::nvidiagpu::Shape reduce_axis,
                                          const char* op,
                                          const char* kernel);
                size_t build_reduce_to_scalar(const std::vector<element::Type>& dtypes,
                                              runtime::nvidiagpu::Shape input_shape,
                                              const char* op,
                                              const char* kernel);
                /// \brief This is the preprocess for reduce to scalar if the data size is large than a number.
                /// The number can be tuned based on hardware.
                /// This cuda kernel will accumulate reduction to a certain number of bins depends on hardware.
                size_t build_reduce_to_scalar_acc(const std::vector<element::Type>& dtypes,
                                                  runtime::nvidiagpu::Shape input_shape,
                                                  runtime::nvidiagpu::Shape output_shape,
                                                  uint32_t block_size_x,
                                                  const char* op,
                                                  const char* kernel);
                /// \brief Simplifed reduce shape and reduce axis, remove dimsion size 1,
                /// combine two or more adjacent reduce/nonreduce axis.
                /// the simplified reduce shape and reduce axis will make index caculation simplier in cuda kernel.
                /// example:
                /// {1 1 2 2} with reduce axis {3} simplifiy to: {2 2} with reduce_axis {1};
                /// {2 3 4} with reduce axis {0 1} simplify to {6 4} with reduce_axis {0};
                /// {2 3 4} with reduce axis {0} simplify to {2 12} with reduce_axis {0};
                void simplify_reduce_shape(runtime::nvidiagpu::Shape in,
                                           runtime::nvidiagpu::Shape reduce_axis,
                                           runtime::nvidiagpu::Shape& simplified_shape,
                                           runtime::nvidiagpu::Shape& simplified_reduce_axis);
                /// \brief Seperate input_shape to reduced_shape and non_reduce_shape, and calcuate strides for them
                ///        and strides in input. This help caculate input index and output index for cuda kernel.
                /// example:
                /// input_shape {2 3 4 5} with reduce_axis {0 2}:
                /// input_strides: {60, 20, 5, 1}
                /// reduce_shape {2 4}, reduce_strides {4 1}, reduce_strides_in_input {60 5}
                /// non_reduce_shape {3 5}, non_reduce_strides {5 1}, non_reduce_strides_in_input {20 1}
                void get_reduce_strides(runtime::nvidiagpu::Shape input_shape,
                                        runtime::nvidiagpu::Shape reduce_axis,
                                        runtime::nvidiagpu::Shape& non_reduce_shape,
                                        runtime::nvidiagpu::Shape& non_reduce_strides,
                                        runtime::nvidiagpu::Shape& non_reduce_strides_in_input,
                                        runtime::nvidiagpu::Shape& reduce_shape,
                                        runtime::nvidiagpu::Shape& reduce_strides,
                                        runtime::nvidiagpu::Shape& reduce_strides_in_input);

                /// \brief Calculate magic and shift part of an shape vector (denomitor), change divide to multiply
                ///        in cuda kernel.
                void div_to_mul(const runtime::nvidiagpu::Shape& shape,
                                std::vector<int>& magic,
                                std::vector<int>& shift);
                /// \brief Get initial value for reduce op
                void* get_init_reduce_val(std::string reduce_op, std::string data_type);
                /// \brief Get vector<string> of datatype from vector<element::Type>
                std::vector<std::string>
                    get_string_vector(const std::vector<element::Type>& dtypes);

                std::shared_ptr<HostParameters> m_host_parameters;
                PrimitiveEmitter* m_primitive_emitter;
                RuntimeContext* m_ctx;
            };
        }
    }
}
