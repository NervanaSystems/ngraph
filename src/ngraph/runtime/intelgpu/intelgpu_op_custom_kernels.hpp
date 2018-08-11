/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include <CPP/topology.hpp>

#include "ngraph/runtime/intelgpu/code_writer.hpp"

#include "ngraph/axis_set.hpp"
#include "ngraph/coordinate.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/strides.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace intelgpu
        {
            void do_pad_operation(cldnn::topology& topology,
                                  const std::string& input_name,
                                  const Shape& input_shape,
                                  const std::string& scalar_name,
                                  const std::string& output_name,
                                  const Shape& output_shape,
                                  const element::Type& output_type,
                                  const Shape& pad_below,
                                  const Shape& pad_interior);

            void do_dot_operation(cldnn::topology& topology,
                                  const std::string& inputA_name,
                                  const Shape& inputA_shape,
                                  const std::string& inputB_name,
                                  const Shape& inputB_shape,
                                  const std::string& output_name,
                                  const Shape& output_shape,
                                  const element::Type& output_type);

            void do_slice_operation(cldnn::topology& topology,
                                    const std::string& input_name,
                                    const Shape& input_shape,
                                    const std::string& output_name,
                                    const Shape& output_shape,
                                    const element::Type& output_type,
                                    const Coordinate& lower_bounds,
                                    const Coordinate& uppper_bounds,
                                    const Strides& strides);

            void do_select_operation(cldnn::topology& topology,
                                     const std::string& input0_name,
                                     const Shape& input0_shape,
                                     const std::string& input1_name,
                                     const Shape& input1_shape,
                                     const std::string& input2_name,
                                     const Shape& input2_shape,
                                     const std::string& output_name,
                                     const Shape& output_shape,
                                     const element::Type& output_type);

            void do_logic_kernel(cldnn::topology& topology,
                                 const std::string& inputA_name,
                                 const Shape& inputA_shape,
                                 const std::string& inputA_type,
                                 const std::string& inputB_name,
                                 const Shape& inputB_shape,
                                 const std::string& inputB_type,
                                 const std::string& output_name,
                                 const Shape& output_shape,
                                 const element::Type& output_type,
                                 const std::string& operation);

            void do_reverse_operation(cldnn::topology& topology,
                                      const std::string& input_name,
                                      const Shape& input_shape,
                                      const std::string& output_name,
                                      const Shape& output_shape,
                                      const element::Type& output_type,
                                      const AxisSet& reversed_axes);

            void do_convert_operation(cldnn::topology& topology,
                                      const std::string& input_name,
                                      const Shape& input_shape,
                                      const element::Type& input_type,
                                      const std::string& output_name,
                                      const Shape& output_shape,
                                      const element::Type& output_type);

            // Helper functions used in cldnn::custom_gpu_primitive kernels
            std::vector<cldnn_arg> get_kernel_args(size_t input, size_t output);
            std::string array_dims(const Shape& dimentions);
            std::string access_dims(const Shape& dimentions,
                                    const AxisSet& axis = {},
                                    bool is_reversed = false);
            std::vector<size_t>
                generate_loops(codegen::CodeWriter& writer, const Shape& shape, bool is_begin);
        }
    }
}
