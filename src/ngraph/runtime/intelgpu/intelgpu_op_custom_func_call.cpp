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

#include <CPP/custom_gpu_primitive.hpp>

#include "ngraph/code_writer.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_layout.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_custom_func_call.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_custom_kernels.hpp"

using namespace std;
using namespace ngraph;

void runtime::intelgpu::do_all_any_op(cldnn::topology& topology,
                                      const string& input0_name,
                                      const Shape& input0_shape,
                                      const string& output_name,
                                      const Shape& output_shape,
                                      const element::Type& output_type,
                                      const AxisSet& axis,
                                      const std::string& operation,
                                      const std::string& init_val)
{
    const string entry_point_name = "custom_op_all_any_" + output_name;
    const string kernel_type_name = get_opencl_type_name(output_type);
    const size_t input_size = shape_size<Shape>(input0_shape);
    CodeWriter writer;

    // The kernel name and parameters
    gen_func_def(writer,
                 entry_point_name,
                 {1, kernel_type_name},
                 {input0_shape, {1}},
                 kernel_type_name,
                 output_shape);

    writer.block_begin();
    {
        // Initialization loop
        size_t var_idx = 0;
        for (auto const& i : output_shape)
        {
            writer << "for (uint i" << var_idx << " = 0; i" << var_idx << " < " << i << "; ++i"
                   << var_idx << ")\n";
            writer.block_begin();
            ++var_idx;
        }

        writer << "output" << access_dims(output_shape) << " = " << init_val << ";\n";

        // Closing brackets for initialization loop
        for (auto const& i : output_shape)
        {
            writer.block_end();
        }

        if (input_size && !input0_shape.empty())
        {
            // Main operation loop
            var_idx = 0;
            for (auto const& i : input0_shape)
            {
                writer << "for (uint i" << var_idx << " = 0; i" << var_idx << " < " << i << "; ++i"
                       << var_idx << ")\n";
                writer.block_begin();
                ++var_idx;
            }

            writer << kernel_type_name << " lhs = output" << access_dims(input0_shape, "i", axis)
                   << ";\n"
                   << kernel_type_name << " rhs = input0" << access_dims(input0_shape) << ";\n"
                   << "output" << access_dims(input0_shape, "i", axis) << " = (" << operation
                   << ");\n";

            // Closing brackets for loop
            for (auto const& i : input0_shape)
            {
                writer.block_end();
            }
        }
    } // End of function bracket
    writer.block_end();

    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const cldnn::custom_gpu_primitive op_all_any(output_name,
                                                 {input0_name},
                                                 {writer.get_code()},
                                                 entry_point_name,
                                                 get_kernel_args(1, 1),
                                                 "",
                                                 layout,
                                                 {1});
    topology.add(op_all_any);
}
