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

#include <CPP/concatenation.hpp>
#include <CPP/custom_gpu_primitive.hpp>
#include <CPP/reshape.hpp>

#include "ngraph/runtime/intelgpu/code_writer.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_layout.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_broadcast.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_custom_kernels.hpp"

#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

void runtime::intelgpu::do_bcast_sum_operation(cldnn::topology& topology,
                                               const string& input_name,
                                               const Shape& input_shape,
                                               const element::Type& input_type,
                                               const string& output_name,
                                               const Shape& output_shape,
                                               const element::Type& output_type,
                                               const AxisSet& axis,
                                               bool is_bcast)
{
    string function_name = is_bcast ? "broadcast_" : "sum_";
    function_name += output_name;
    codegen::CodeWriter writer;
    vector<size_t> gws;

    runtime::intelgpu::gen_func_def(writer,
                                    function_name,
                                    {get_opencl_type_name(input_type)},
                                    {input_shape},
                                    get_opencl_type_name(output_type),
                                    output_shape);
    writer.block_begin();
    {
        if (is_bcast)
        {
            // Broadcast loops
            gws = runtime::intelgpu::generate_loops(writer, output_shape, true);

            writer << "output" << access_dims(output_shape) << " = input0"
                   << access_dims(output_shape, "i", axis) << ";\n";

            // Closing brackets for Broadcast loop
            runtime::intelgpu::generate_loops(writer, output_shape, false);
        }
        else
        {
            const string reduction_str =
                "output" + access_dims(input_shape, "i", axis) + " = result;\n";

            // Generate loops related to input order with GWS
            gws = generate_loops_w_axes(writer,
                                        input_shape,
                                        true,
                                        axis,
                                        get_opencl_type_name(output_type) + " result = 0.0f;\n");

            writer << "result += input0" << access_dims(input_shape) << ";\n";

            // Close brackets related to input order with reduction
            generate_loops_w_axes(writer, input_shape, false, axis, reduction_str);
        }
    } // End of function bracket
    writer.block_end();

    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const cldnn::custom_gpu_primitive op_bcast_sum(output_name,
                                                   {input_name},
                                                   {writer.get_code()},
                                                   function_name,
                                                   get_kernel_args(1, 1),
                                                   "",
                                                   layout,
                                                   gws);
    topology.add(op_bcast_sum);
}

void runtime::intelgpu::do_max_min_operation(cldnn::topology& topology,
                                             const string& input_name,
                                             const Shape& input_shape,
                                             const string& output_name,
                                             const Shape& output_shape,
                                             const element::Type& output_type,
                                             const AxisSet& axis,
                                             bool is_min)
{
    const string function_name = "min_max_" + output_name;
    const size_t input_size = shape_size<Shape>(input_shape);
    const string& init_value = is_min ? "INFINITY" : "-INFINITY";
    const string& operation = is_min ? " < " : " > ";
    codegen::CodeWriter writer;

    writer << "__kernel void " << function_name << "(const __global float input"
           << array_dims(input_shape) << ", __global float output" << array_dims(output_shape)
           << ")\n";

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

        writer << "output" << access_dims(output_shape) << " = " << init_value << ";\n";

        // Closing brackets for initialization loop
        for (auto const& i : output_shape)
        {
            writer.block_end();
        }

        if (input_size && !input_shape.empty())
        {
            // Main operation loop
            var_idx = 0;
            for (auto const& i : input_shape)
            {
                writer << "for (uint i" << var_idx << " = 0; i" << var_idx << " < " << i << "; ++i"
                       << var_idx << ")\n";
                writer.block_begin();
                ++var_idx;
            }

            writer << "if (input" << access_dims(input_shape) << operation << "output"
                   << access_dims(input_shape, "i", axis) << ")\n";
            writer.block_begin();
            {
                writer << "output" << access_dims(input_shape, "i", axis) << " = input"
                       << access_dims(input_shape) << ";\n";
            }
            writer.block_end();

            // Closing brackets for loop
            for (auto const& i : input_shape)
            {
                writer.block_end();
            }
        }
    } // End of function bracket
    writer.block_end();

    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const cldnn::custom_gpu_primitive op_min_max(output_name,
                                                 {input_name},
                                                 {writer.get_code()},
                                                 function_name,
                                                 get_kernel_args(1, 1),
                                                 "",
                                                 layout,
                                                 {1});
    topology.add(op_min_max);
}

void runtime::intelgpu::do_product_operation(cldnn::topology& topology,
                                             const string& input_name,
                                             const Shape& input_shape,
                                             const string& output_name,
                                             const Shape& output_shape,
                                             const element::Type& output_type,
                                             const AxisSet& axis)
{
    const string function_name = "product_" + output_name;
    const size_t input_size = shape_size<Shape>(input_shape);
    codegen::CodeWriter writer;

    writer << "__kernel void " << function_name << "(const __global float input"
           << array_dims(input_shape) << ", __global float output" << array_dims(output_shape)
           << ")\n";

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

        writer << "output" << access_dims(output_shape) << " = 1;\n";

        // Closing brackets for initialization loop
        for (auto const& i : output_shape)
        {
            writer.block_end();
        }

        if (input_size && !input_shape.empty())
        {
            // Main operation loop
            var_idx = 0;
            for (auto const& i : input_shape)
            {
                writer << "for (uint i" << var_idx << " = 0; i" << var_idx << " < " << i << "; ++i"
                       << var_idx << ")\n";
                writer.block_begin();
                ++var_idx;
            }

            writer << "output" << access_dims(input_shape, "i", axis) << " *= input"
                   << access_dims(input_shape) << ";\n";

            // Closing brackets for loop
            for (auto const& i : input_shape)
            {
                writer.block_end();
            }
        }
    } // End of function bracket
    writer.block_end();

    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const cldnn::custom_gpu_primitive op_product(output_name,
                                                 {input_name},
                                                 {writer.get_code()},
                                                 function_name,
                                                 get_kernel_args(1, 1),
                                                 "",
                                                 layout,
                                                 {1});
    topology.add(op_product);
}
