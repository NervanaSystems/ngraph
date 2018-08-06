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

void runtime::intelgpu::do_bcast_sum_operation_scalar(cldnn::topology& topology,
                                                      const string& input_name,
                                                      const Shape& input_shape,
                                                      const string& output_name,
                                                      const Shape& output_shape,
                                                      const element::Type& output_type,
                                                      bool is_bcast)
{
    string function_name = is_bcast ? "broadcast_scalar" : "sum_scalar";
    function_name += output_name;
    const size_t input_count =
        is_bcast ? shape_size<Shape>(output_shape) : shape_size<Shape>(input_shape);
    codegen::CodeWriter writer;

    writer << "__kernel void " << function_name
           << "(const __global float* input, __global float* output)\n";
    writer.block_begin();
    {
        writer << "float sum = 0.f;\n"
               << "for (uint i = 0; i < COUNT; ++i)\n";
        writer.block_begin();

        if (is_bcast)
        {
            writer << "output[i] = input[0];\n";
            writer.block_end();
        }
        else
        {
            writer << "sum += input[i];\n";
            writer.block_end();
            writer << "output[0] = sum;\n";
        }
    } // End of function bracket
    writer.block_end();

    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const cldnn::custom_gpu_primitive op_scalar(output_name,
                                                {input_name},
                                                {writer.get_code()},
                                                function_name,
                                                get_kernel_args(1, 1),
                                                string("-DCOUNT=" + to_string(input_count)),
                                                layout,
                                                {1});
    topology.add(op_scalar);
}

void runtime::intelgpu::do_bcast_sum_operation(cldnn::topology& topology,
                                               const string& input_name,
                                               const Shape& input_shape,
                                               const string& output_name,
                                               const Shape& output_shape,
                                               const element::Type& output_type,
                                               const AxisSet& axis,
                                               bool is_bcast)
{
    const string function_name = is_bcast ? "broadcast" : "sum";
    codegen::CodeWriter writer;

    writer << "__kernel void " << function_name << "(const __global float input"
           << array_dims(input_shape) << ", __global float output" << array_dims(output_shape)
           << ")\n";

    writer.block_begin();
    {
        if (is_bcast)
        {
            size_t var_idx = 0;
            for (auto const& i : output_shape)
            {
                writer << "for (uint i" << var_idx << " = 0; i" << var_idx << " < " << i << "; ++i"
                       << var_idx << ")\n";
                writer.block_begin();
                ++var_idx;
            }
            writer << "output" << access_dims(output_shape) << " = input"
                   << access_dims(output_shape, axis) << ";\n";

            // Closing brackets for Broadcast loop
            for (auto const& i : output_shape)
            {
                writer.block_end();
            }
        }
        else
        {
            // Initialize destination output by zeroes
            size_t var_idx = 0;
            for (auto const& i : output_shape)
            {
                writer << "for (uint i" << var_idx << " = 0; i" << var_idx << " < " << i << "; ++i"
                       << var_idx << ")\n";
                writer.block_begin();
                ++var_idx;
            }

            writer << "output" << access_dims(output_shape) << " = 0.0f;\n";

            // Closing brackets for Sum initialization loop
            for (auto const& i : output_shape)
            {
                writer.block_end();
            }

            // Now do the Sum operation
            var_idx = 0;
            for (auto const& i : input_shape)
            {
                writer << "for (uint i" << var_idx << " = 0; i" << var_idx << " < " << i << "; ++i"
                       << var_idx << ")\n";
                writer.block_begin();
                ++var_idx;
            }

            writer << "output" << access_dims(input_shape, axis) << " += input"
                   << access_dims(input_shape) << ";\n";

            // Closing brackets for Sum loop
            for (auto const& i : input_shape)
            {
                writer.block_end();
            }
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
                                                   {1});
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
                   << access_dims(input_shape, axis) << ")\n";
            writer.block_begin();
            {
                writer << "output" << access_dims(input_shape, axis) << " = input"
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

            writer << "output" << access_dims(input_shape, axis) << " *= input"
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
