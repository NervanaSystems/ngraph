//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "ngraph/op/pad.hpp"

namespace ngraph
{
    class AxisSet;
    class AxisVector;
    class Coordinate;
    class CoordinateDiff;
    class Shape;
    class Strides;

    namespace runtime
    {
        namespace cpu
        {
            namespace kernel
            {
                void pad_4d_float32(float* input,
                                    float* output,
                                    float* pad_value,
                                    const Shape& input_shape,
                                    const Shape& output_shape,
                                    const CoordinateDiff& padding_below,
                                    const CoordinateDiff& padding_above,
                                    const ngraph::op::PadMode pad_mode,
                                    int arena);

                void reduce_sum_all_1d_float32(float* input,
                                               float* output,
                                               const Shape& input_shape,
                                               const Shape& output_shape,
                                               int arena);

                void reduce_sum_all_2d_float32(float* input,
                                               float* output,
                                               const Shape& input_shape,
                                               const Shape& output_shape,
                                               int arena);

                void reduce_sum_2d_1rd_float32(float* input,
                                               float* output,
                                               const Shape& input_shape,
                                               const Shape& output_shape,
                                               const AxisSet& reduction_axes,
                                               int arena);

                void reduce_sum_4d_2rd_float32(float* input,
                                               float* output,
                                               const Shape& input_shape,
                                               const Shape& output_shape,
                                               const AxisSet& reduction_axes,
                                               int arena);

                void reduce_sum_all_4d_float32(float* input,
                                               float* output,
                                               const Shape& input_shape,
                                               const Shape& output_shape,
                                               int arena);

                void reduce_max_2d_1rd_float32(float* input,
                                               float* output,
                                               const Shape& input_shape,
                                               const Shape& output_shape,
                                               const AxisSet& reduction_axes,
                                               int arena);

                void reshape_3d_3d_float32(float* input,
                                           float* output,
                                           const Shape& input_shape,
                                           const AxisVector& input_axis_order,
                                           const Shape& output_shape,
                                           int arena);

                void reshape_4d_4d_float32(float* input,
                                           float* output,
                                           const Shape& input_shape,
                                           const AxisVector& input_axis_order,
                                           const Shape& output_shape,
                                           int arena);

                template <typename ElementType, unsigned int Rank>
                void update_slice(void* input0,
                                  void* input1,
                                  void* output,
                                  const Shape& input0_shape,
                                  const Shape& input1_shape,
                                  const Coordinate& lower_bounds,
                                  int arena);

                template <typename ElementType, unsigned int Rank>
                void strided_update_slice(void* input0,
                                          void* input1,
                                          void* output,
                                          const Shape& input0_shape,
                                          const Shape& input1_shape,
                                          const Coordinate& lower_bounds,
                                          const Coordinate& upper_bounds,
                                          const Strides& slice_strides,
                                          int arena);

                template <typename ElementType>
                void erf(void* input0, void* output, size_t count, int arena);

                template <typename ElementType>
                void reference_erf(void* arg, void* out, size_t count);

                template <typename ElementType>
                void tile_rank_0(void* input, void* output, size_t repeats);

                template <typename ElementType, unsigned int Rank>
                void tile(void* input,
                          void* output,
                          const Shape& input_shape,
                          const Shape& output_shape,
                          int arena);

                template <typename ElementType,
                          typename IndicesType,
                          unsigned int Rank1,
                          unsigned int Rank2>
                void gather(void* inputs,
                            void* indices,
                            void* output,
                            const Shape& inputs_shape,
                            const Shape& indices_shape,
                            const Shape& output_shape,
                            size_t axis,
                            int arena);

                template <typename ElementType,
                          typename IndicesType,
                          unsigned int Rank1,
                          unsigned int Rank2>
                void scatter_add(void* inputs,
                                 void* indices,
                                 void* updates,
                                 void* output,
                                 const Shape& inputs_shape,
                                 const Shape& indices_shape,
                                 const Shape& updates_shape,
                                 int arena);

                template <typename T>
                void generate_dropout(T* input,
                                      T* out0,
                                      T* out1_mask,
                                      size_t nelems,
                                      bool training,
                                      const double value,
                                      const std::vector<std::minstd_rand>& vmsr,
                                      const bool use_seed);

                template <typename InputElementType, typename AxisElementType>
                void reference_cumsum(void* input_tensor,
                                      void* axis_tensor,
                                      void* out,
                                      const Shape& tensor_shape,
                                      const bool exclusive,
                                      const bool reverse);
            }
        }
    }
}
