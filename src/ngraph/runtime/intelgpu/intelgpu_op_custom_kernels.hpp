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

#include <CPP/topology.hpp>

#include "ngraph/code_writer.hpp"

#include "ngraph/axis_set.hpp"
#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/strides.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace intelgpu
        {
            size_t get_max_memory_rss();

            void do_pad_operation(cldnn::topology& topology,
                                  const std::string& input_name,
                                  const Shape& input_shape,
                                  const std::string& scalar_name,
                                  const std::string& output_name,
                                  const Shape& output_shape,
                                  const element::Type& output_type,
                                  const CoordinateDiff& pad_below);

            void do_max_pool_backprop_operation(cldnn::topology& topology,
                                                const std::string& input_name,
                                                const Shape& input_shape,
                                                const std::string& delta_name,
                                                const Shape& delta_shape,
                                                const std::string& output_name,
                                                const Shape& output_shape,
                                                const element::Type& output_type,
                                                const Shape& win_shape,
                                                const Shape& win_stride,
                                                const Shape& pad_below);

            void do_max_avg_pool_operation(cldnn::topology& topology,
                                           const std::string& input_name,
                                           const Shape& input_shape,
                                           const std::string& output_name,
                                           const Shape& output_shape,
                                           const element::Type& output_type,
                                           const Shape& win_shape,
                                           const Shape& win_stride,
                                           const Shape& pad_below,
                                           bool include_padding,
                                           const std::string& def_val,
                                           bool is_max_pool);

            void do_avg_pool_backprop_operation(cldnn::topology& topology,
                                                const std::string& delta_name,
                                                const Shape& delta_shape,
                                                const std::string& output_name,
                                                const Shape& output_shape,
                                                const element::Type& output_type,
                                                const Shape& win_shape,
                                                const Shape& win_stride,
                                                const Shape& pad_below,
                                                const bool include_padding);

            void do_dot_operation(cldnn::topology& topology,
                                  const std::string& inputA_name,
                                  const Shape& inputA_shape,
                                  const std::string& inputB_name,
                                  const Shape& inputB_shape,
                                  const std::string& output_name,
                                  const Shape& output_shape,
                                  const element::Type& output_type,
                                  size_t reduction_axes_count);

            void do_concat_operation(cldnn::topology& topology,
                                     const std::vector<std::string>& input_names,
                                     const std::vector<Shape>& input_shapes,
                                     const std::string& output_name,
                                     const Shape& output_shape,
                                     const element::Type& output_type,
                                     size_t concat_axis);

            void do_logic_kernel(cldnn::topology& topology,
                                 const std::string& input0_name,
                                 const Shape& input0_shape,
                                 const element::Type& input0_type,
                                 const std::string& input1_name,
                                 const Shape& input1_shape,
                                 const std::string& output_name,
                                 const Shape& output_shape,
                                 const element::Type& output_type,
                                 const std::string& operation);

            void do_eltwise_kernel(cldnn::topology& topology,
                                   const std::string& input0_name,
                                   const Shape& input0_shape,
                                   const element::Type& input0_type,
                                   const std::string& input1_name,
                                   const Shape& input1_shape,
                                   const std::string& output_name,
                                   const Shape& output_shape,
                                   const element::Type& output_type,
                                   const std::string& operation,
                                   bool function_operation);

            void do_relu_backprop(cldnn::topology& topology,
                                  const std::string& input0_name,
                                  const Shape& input0_shape,
                                  const element::Type& input0_type,
                                  const std::string& input1_name,
                                  const Shape& input1_shape,
                                  const std::string& output_name,
                                  const Shape& output_shape,
                                  const element::Type& output_type);

            void do_reverse_operation(cldnn::topology& topology,
                                      const std::string& input_name,
                                      const Shape& input_shape,
                                      const element::Type& input_type,
                                      const std::string& output_name,
                                      const Shape& output_shape,
                                      const element::Type& output_type,
                                      const AxisSet& reversed_axes);

            void do_reverse_sequence_operation(cldnn::topology& topology,
                                               const std::string& input0_name,
                                               const Shape& input0_shape,
                                               const element::Type& input0_type,
                                               const std::string& input1_name,
                                               const Shape& input1_shape,
                                               const element::Type& input1_type,
                                               const std::string& output_name,
                                               const Shape& output_shape,
                                               const element::Type& output_type,
                                               const size_t reversed_axis,
                                               const size_t batch_axis);

            void do_not_operation(cldnn::topology& topology,
                                  const std::string& input_name,
                                  const Shape& input_shape,
                                  const std::string& output_name,
                                  const Shape& output_shape,
                                  const element::Type& output_type);

            void do_one_hot_operation(cldnn::topology& topology,
                                      const std::string& input_name,
                                      const Shape& input_shape,
                                      const element::Type& input_type,
                                      const std::string& output_name,
                                      const Shape& output_shape,
                                      const element::Type& output_type,
                                      const size_t one_hot_axis);

            void do_convert_operation(cldnn::topology& topology,
                                      const std::string& input_name,
                                      const Shape& input_shape,
                                      const element::Type& input_type,
                                      const std::string& output_name,
                                      const Shape& output_shape,
                                      const element::Type& output_type);

            void do_sigmoid_backprop_operation(cldnn::topology& topology,
                                               const std::string& input_name,
                                               const Shape& input_shape,
                                               const std::string& delta_name,
                                               const Shape& delta_shape,
                                               const std::string& output_name,
                                               const Shape& output_shape,
                                               const element::Type& output_type);

            void do_custom_unary_operation(cldnn::topology& topology,
                                           const std::string& input_name,
                                           const Shape& input_shape,
                                           const element::Type& input_type,
                                           const std::string& output_name,
                                           const Shape& output_shape,
                                           const element::Type& output_type,
                                           const std::string& operation_name);

            void do_arg_max_min_operation(cldnn::topology& topology,
                                          const std::string& input_name,
                                          const Shape& input_shape,
                                          const element::Type& input_type,
                                          const std::string& output_name,
                                          const Shape& output_shape,
                                          const element::Type& output_type,
                                          const size_t reduction_axis,
                                          const bool is_max);

            void do_reshape_operation(cldnn::topology& topology,
                                      const std::string& input_name,
                                      const Shape& input_shape,
                                      const element::Type& input_type,
                                      const std::string& output_name,
                                      const Shape& output_shape,
                                      const element::Type& output_type,
                                      const AxisVector& reshape_axes);

            void do_quantize_operation(cldnn::topology& topology,
                                       const std::string& input0_name,
                                       const Shape& input0_shape,
                                       const element::Type& input0_type,
                                       const std::string& input1_name,
                                       const Shape& input1_shape,
                                       const std::string& input2_name,
                                       const Shape& input2_shape,
                                       const std::string& output_name,
                                       const Shape& output_shape,
                                       const element::Type& output_type,
                                       const AxisSet& axis,
                                       const ngraph::op::Quantize::RoundMode mode);

            void do_dequantize_operation(cldnn::topology& topology,
                                         const std::string& input0_name,
                                         const Shape& input0_shape,
                                         const element::Type& input0_type,
                                         const std::string& input1_name,
                                         const Shape& input1_shape,
                                         const element::Type& input1_type,
                                         const std::string& input2_name,
                                         const Shape& input2_shape,
                                         const element::Type& input2_type,
                                         const std::string& output_name,
                                         const Shape& output_shape,
                                         const element::Type& output_type,
                                         const AxisSet& axis);

            void do_topk_operation(cldnn::topology& topology,
                                   const std::string& input_name,
                                   const Shape& input_shape,
                                   const element::Type& input_type,
                                   const std::string& output_name,
                                   const Shape& output_shape,
                                   const element::Type& output_type,
                                   const element::Type& index_elem_type,
                                   const size_t top_k_axis,
                                   const size_t k,
                                   const bool compute_max,
                                   const bool find_indices);

            // Helper functions used in cldnn::custom_gpu_primitive kernels
            std::string get_opencl_type_name(const element::Type& ngraph_type);
            std::string get_opencl_type_min_max_value(const element::Type& ngraph_type,
                                                      bool is_min);
            std::vector<cldnn_arg> get_kernel_args(size_t input, size_t output);
            std::string array_dims(const Shape& dimentions, const AxisSet& axis = {});
            std::string access_dims(const Shape& dimentions,
                                    const std::string& var = "i",
                                    const AxisSet& axis = {},
                                    bool is_reversed = false);
            std::vector<size_t>
                generate_loops(CodeWriter& writer, const Shape& shape, bool is_begin);
            std::vector<size_t>
                generate_loops_w_axes(CodeWriter& writer,
                                      const Shape& shape,
                                      bool is_begin,
                                      const AxisSet& axis = {},
                                      const std::string& expression = std::string());
            void gen_func_def(CodeWriter& writer,
                              const std::string& entry_point_name,
                              const std::vector<std::string>& input_types,
                              const std::vector<Shape>& input_shapes,
                              const std::string& output_type,
                              const Shape& output_shape);
        }
    }
}
