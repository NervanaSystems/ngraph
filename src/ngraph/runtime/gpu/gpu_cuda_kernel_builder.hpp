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
#include <set>
#include <string>
#include <vector>

namespace ngraph
{
    namespace codegen
    {
        class CodeWriter;
    }
    namespace runtime
    {
        namespace gpu
        {
            class GPUKernelArgs;
            class CudaKernelBuilder
            {
            public:
                static void get_elementwise_op(CodeWriter& writer,
                                               const std::string& name,
                                               const std::string& op,
                                               const std::vector<std::string>& data_types);

                static void get_memset_op(CodeWriter& writer,
                                          const std::string& name,
                                          const std::string& data_type,
                                          runtime::gpu::GPUKernelArgs& args);

                static void get_cudnn_bn_inv_var_op(CodeWriter& writer,
                                                    const std::string& name,
                                                    runtime::gpu::GPUKernelArgs& args);

                static void get_broadcast_op(CodeWriter& writer,
                                             const std::string& name,
                                             const std::string& data_type,
                                             GPUKernelArgs& args,
                                             const size_t rank);

                static void get_concat_op(CodeWriter& writer,
                                          const std::string& name,
                                          const std::string& data_type,
                                          size_t num_inputs);

                static void get_onehot_op(CodeWriter& writer,
                                          const std::string& name,
                                          const std::array<std::string, 2>& data_types);

                static void get_reshape_op(CodeWriter& writer,
                                           const std::string& name,
                                           runtime::gpu::GPUKernelArgs& args,
                                           const std::array<std::string, 2>& data_types,
                                           size_t rank);

                static void get_reshape_op_3d(CodeWriter& writer,
                                              const std::string& name,
                                              runtime::gpu::GPUKernelArgs& args,
                                              const std::string& data_type,
                                              const std::vector<uint32_t>& order,
                                              const std::vector<uint32_t>& block_size);

                static void get_reshape_op_2d(CodeWriter& writer,
                                              const std::string& name,
                                              runtime::gpu::GPUKernelArgs& args,
                                              const std::string& data_type,
                                              uint32_t block_size);

                /// \brief reduce op for output that is not scalar
                /// stable kahan sum is been used for float point sum.
                /// no initial value needed since we load one input value as initial
                /// not support 0 sized input
                static void get_reduce_to_nd_op(CodeWriter& writer,
                                                const std::string& name,
                                                runtime::gpu::GPUKernelArgs& args,
                                                const std::vector<std::string>& data_types,
                                                const std::string& reduce_op,
                                                size_t non_reduce_rank,
                                                size_t reduce_rank);

                /// \brief This is the preprocess to reduce to scalar if the input data size is
                ///        large than a number.
                /// The number can be tuned based on hardware.
                /// This cuda kernel will accumulate reduction to a certain number of bins depends
                /// on hardware.
                /// stable kahan sum is been used for float point sum.
                /// no initial value needed since we load one input value as initial
                /// not support 0 sized input
                static void get_reduce_to_scalar_acc_op(CodeWriter& writer,
                                                        const std::string& name,
                                                        runtime::gpu::GPUKernelArgs& args,
                                                        const std::vector<std::string>& data_types,
                                                        const std::string& reduce_op);

                /// \brief This op using one block with at most 512 threads to reduce to scalar.
                /// stable kahan sum is been used for float point sum.
                /// no initial value needed since we load one input value as initial
                /// not support 0 sized input
                static void get_reduce_to_scalar_op(CodeWriter& writer,
                                                    const std::string& name,
                                                    runtime::gpu::GPUKernelArgs& args,
                                                    const std::vector<std::string>& data_types,
                                                    const std::string& reduce_op,
                                                    uint32_t block_size_x);

                static void get_topk(CodeWriter& writer,
                                     const std::string& name,
                                     const std::vector<std::string>& dtypes,
                                     bool compute_max,
                                     runtime::gpu::GPUKernelArgs& args,
                                     bool use_malloc);

                static void get_slice_op(CodeWriter& writer,
                                         const std::string& name,
                                         const std::array<std::string, 2>& data_types,
                                         size_t rank);

                static void get_reverse_op(CodeWriter& writer,
                                           const std::string& name,
                                           const std::array<std::string, 2>& data_types);

                static void get_replace_slice_op(CodeWriter& writer,
                                                 const std::string& name,
                                                 GPUKernelArgs& args,
                                                 const size_t rank);

                static void get_reverse_sequence_op(CodeWriter& writer,
                                                    const std::string& name,
                                                    const std::array<std::string, 3>& data_types,
                                                    const size_t batch_axis,
                                                    const size_t sequence_axis,
                                                    const size_t rank);

                static void get_device_helper(CodeWriter& writer,
                                              const std::string& name,
                                              const std::string& math_kernel,
                                              const std::vector<std::string>& data_types);

                static void get_pad_op(CodeWriter& writer,
                                       const std::string& name,
                                       GPUKernelArgs& args,
                                       size_t rank);

                static void get_pad_fill_op(CodeWriter& writer,
                                            const std::string& name,
                                            GPUKernelArgs& args,
                                            size_t rank);

                static void get_ew_collective_op(CodeWriter& writer,
                                                 const std::string& name,
                                                 GPUKernelArgs& args,
                                                 const std::string& op,
                                                 const std::string& reduce_op,
                                                 const std::vector<std::string>& data_types,
                                                 const std::set<size_t>& reduced_tensors,
                                                 bool save_elementwise,
                                                 size_t rank);

                static void get_max_pool_1d(CodeWriter& writer,
                                            const std::string& name,
                                            const std::array<std::string, 2>& data_types,
                                            size_t input_width,
                                            size_t output_width,
                                            size_t window_width,
                                            size_t window_stride);

                static void get_avg_pool(CodeWriter& writer,
                                         const std::string& name,
                                         const std::array<std::string, 2>& data_types,
                                         bool include_pad);

                static void get_convolution_forward(CodeWriter& writer,
                                                    const std::string& name,
                                                    const std::array<std::string, 3>& data_types,
                                                    GPUKernelArgs& args,
                                                    int N,
                                                    int K,
                                                    int rank,
                                                    int filter_size,
                                                    int sm_tile_size = 8,
                                                    int reg_tile_size = 1);

                static void get_softmax_op(CodeWriter& writer,
                                           const std::string& name,
                                           runtime::gpu::GPUKernelArgs& args,
                                           const std::vector<std::string>& data_types,
                                           size_t out_rank,
                                           size_t reduce_rank);

                static void get_softmax_block_reduce_op(CodeWriter& writer,
                                                        const std::string& name,
                                                        runtime::gpu::GPUKernelArgs& args,
                                                        const std::vector<std::string>& data_types,
                                                        size_t non_reduce_rank,
                                                        size_t reduce_rank,
                                                        size_t block_size_x);

                static void add_pod_typedefs(CodeWriter& writer);

                static void coordinate_transform_to_multi_d(CodeWriter& writer,
                                                            std::string i_strides,
                                                            std::string i_stride_magic,
                                                            std::string i_stride_shift,
                                                            std::string i_coord_product,
                                                            std::string o_coordinates,
                                                            size_t rank,
                                                            bool register_arguments = false);

                /// \brief Given kernel input variables i_* produce register variables
                ///        o_coordinates{i} of the non-reduced tensor and return the string name of
                ///        integer index into reduced tensor
                static std::string
                    collective_coordinate_transform_helper(CodeWriter& writer,
                                                           std::string i_thread_index,
                                                           std::string i_strides,
                                                           std::string i_stride_magic,
                                                           std::string i_stride_shift,
                                                           std::string i_reduced_strides,
                                                           std::string o_coordinates,
                                                           size_t rank,
                                                           bool register_arguments = true,
                                                           std::string reduced_idx = "reduced_idx");

                static bool stable_sum_check_helper(const std::string& op,
                                                    const std::string& data_type);
            };
        }
    }
}
