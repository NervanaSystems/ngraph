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

#include <CPP/batch_norm.hpp>
#include <CPP/concatenation.hpp>
#include <CPP/custom_gpu_primitive.hpp>
#include <CPP/scale.hpp>
#include <CPP/split.hpp>

#include "ngraph/runtime/intelgpu/code_writer.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_layout.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_batchnorm.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_custom_kernels.hpp"

#include "ngraph/op/batch_norm.hpp"

using namespace std;
using namespace ngraph;

void runtime::intelgpu::do_create_mean(cldnn::topology& topology,
                                       const string& output_name,
                                       const Shape& output_shape,
                                       const element::Type& output_type,
                                       const string& input_name,
                                       const Shape& input_shape)
{
    if (input_shape.size() < 2 || input_shape.size() > 4)
    {
        throw invalid_argument("intelgpu::do_create_mean_variance() wrong input shapes.");
    }

    // According to the documentation, input data channel is always being axis 1
    // Assumed the second dimension from the left. Example {0, 1, 0, 0} or {0, 1}
    // Also, input data must be at least 2D array
    const size_t channel_axis = 1;
    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const string entry_point_name = "create_mean_" + output_name;
    const size_t output_counts = shape_size<Shape>(input_shape) / input_shape.at(channel_axis);
    codegen::CodeWriter writer;

    writer << "__kernel void " << entry_point_name << "( const __global float input"
           << array_dims(input_shape) << ", __global float output" << array_dims(output_shape)
           << ")\n";

    writer.block_begin();
    { // Main function body

        // Loop for Channel axis 1
        writer << "for (uint i" << channel_axis << " = 0; i" << channel_axis << " < "
               << input_shape.at(channel_axis) << "; ++i" << channel_axis << ")\n";
        writer.block_begin();
        {
            writer << "float sum = 0.0f;\n";
            size_t var_idx = 0;
            // Main loops
            for (auto const& i : input_shape)
            {
                if (var_idx != channel_axis)
                {
                    writer << "for (uint i" << var_idx << " = 0; i" << var_idx << " < " << i
                           << "; ++i" << var_idx << ")\n";
                    writer.block_begin();
                }
                ++var_idx;
            }

            writer << "sum += input" << access_dims(input_shape) << ";\n";

            var_idx = 0;
            // Closing brackets for main loops
            for (auto const& i : input_shape)
            {
                if (var_idx != channel_axis)
                {
                    writer.block_end();
                }
                ++var_idx;
            }
            writer << "output[i" << channel_axis << "]  = sum / " << output_counts << ";\n";

        } // Closing brackets for Channel axis loop
        writer.block_end();

    } // Main function body
    writer.block_end();

    const cldnn::custom_gpu_primitive op_mean(output_name,
                                              {input_name},
                                              {writer.get_code()},
                                              entry_point_name,
                                              get_kernel_args(1, 1),
                                              "",
                                              layout,
                                              {1});
    topology.add(op_mean);
}

void runtime::intelgpu::do_create_variance(cldnn::topology& topology,
                                           const string& output_name,
                                           const Shape& output_shape,
                                           const element::Type& output_type,
                                           const string& input_name,
                                           const Shape& input_shape,
                                           const std::string& mean_name)
{
    if (input_shape.size() < 2 || input_shape.size() > 4)
    {
        throw invalid_argument("intelgpu::do_create_mean_variance() wrong input shapes.");
    }

    // According to the documentation, input data channel is always being axis 1
    // Assumed the second dimension from the left. Example {0, 1, 0, 0} or {0, 1}
    // Also, input data must be at least 2D array
    const size_t channel_axis = 1;
    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const string entry_point_name = "create_variance_" + output_name;
    const size_t output_counts = shape_size<Shape>(input_shape) / input_shape.at(channel_axis);
    codegen::CodeWriter writer;

    writer << "__kernel void " << entry_point_name << "( const __global float input"
           << array_dims(input_shape) << ", const __global float mean" << array_dims(output_shape)
           << ", __global float output" << array_dims(output_shape) << ")\n";

    writer.block_begin();
    { // Main function body

        // Loop for Channel axis 1
        writer << "for (uint i" << channel_axis << " = 0; i" << channel_axis << " < "
               << input_shape.at(channel_axis) << "; ++i" << channel_axis << ")\n";
        writer.block_begin();
        {
            writer << "float sum = 0.0f;\n";

            size_t var_idx = 0;
            // Main loops
            for (auto const& i : input_shape)
            {
                if (var_idx != channel_axis)
                {
                    writer << "for (uint i" << var_idx << " = 0; i" << var_idx << " < " << i
                           << "; ++i" << var_idx << ")\n";
                    writer.block_begin();
                }
                ++var_idx;
            }

            writer << "float mean_diff = input" << access_dims(input_shape) << " - mean[i"
                   << channel_axis << "];\n";
            writer << "sum += mean_diff * mean_diff;\n";

            var_idx = 0;
            // Closing brackets for main loops
            for (auto const& i : input_shape)
            {
                if (var_idx != channel_axis)
                {
                    writer.block_end();
                }
                ++var_idx;
            }

            writer << "output[i" << channel_axis << "]  = sum / " << output_counts << ";\n";

        } // Closing brackets for Channel axis loop
        writer.block_end();

    } // Main function body
    writer.block_end();

    const cldnn::custom_gpu_primitive op_variance(output_name,
                                                  {input_name, mean_name},
                                                  {writer.get_code()},
                                                  entry_point_name,
                                                  get_kernel_args(2, 1),
                                                  "",
                                                  layout,
                                                  {1});
    topology.add(op_variance);
}

void runtime::intelgpu::do_batch_norm_operation(cldnn::topology& topology,
                                                const string& output_name,
                                                const Shape& output_shape,
                                                const element::Type& output_type,
                                                double eps,
                                                const string& input_name,
                                                const Shape& input_shape,
                                                const string& gamma_name,
                                                const Shape& gamma_shape,
                                                const string& beta_name,
                                                const string& mean_name_inp,
                                                const string& variance_name_inp)
{
    if (input_shape.size() < 2 || input_shape.size() > 4)
    {
        throw invalid_argument("intelgpu::do_batch_norm_operation() wrong input shapes.");
    }

    // According to the documentation, input data channel is always being axis 1
    // Assumed the second dimension from the left. Example {0, 1, 0, 0} or {0, 1}
    // Also, input data must be at least 2D array
    const size_t channel_axis = 1;
    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const string entry_point_name = "batch_norm_" + output_name;
    codegen::CodeWriter writer;

    writer << "__kernel void " << entry_point_name << "( const __global float input"
           << array_dims(input_shape) << ", const __global float gamma" << array_dims(gamma_shape)
           << ", const __global float beta" << array_dims(gamma_shape)
           << ", const __global float mean" << array_dims(gamma_shape)
           << ", const __global float variance" << array_dims(gamma_shape)
           << ", __global float output" << array_dims(output_shape) << ")\n";

    writer.block_begin();
    { // Main function body

        // Loop for Channel axis 1
        writer << "for (uint i" << channel_axis << " = 0; i" << channel_axis << " < "
               << output_shape.at(channel_axis) << "; ++i" << channel_axis << ")\n";
        writer.block_begin();
        {
            size_t var_idx = 0;
            // Main loops
            for (auto const& i : output_shape)
            {
                if (var_idx != channel_axis)
                {
                    writer << "for (uint i" << var_idx << " = 0; i" << var_idx << " < " << i
                           << "; ++i" << var_idx << ")\n";
                    writer.block_begin();
                }
                ++var_idx;
            }

            writer << "float normalized = (input" << access_dims(input_shape) << " - mean[i"
                   << channel_axis << "]) / ("
                   << "sqrt(variance[i" << channel_axis << "] + " << eps << ")"
                   << ");\n";

            writer << "output" << access_dims(output_shape) << " = normalized * gamma[i"
                   << channel_axis << "] + beta[i" << channel_axis << "];\n";

            var_idx = 0;
            // Closing brackets for main loops
            for (auto const& i : output_shape)
            {
                if (var_idx != channel_axis)
                {
                    writer.block_end();
                }
                ++var_idx;
            }

        } // Closing brackets for Channel axis loop
        writer.block_end();

    } // Main function body
    writer.block_end();

    const vector<cldnn::primitive_id>& inputs = {
        input_name, gamma_name, beta_name, mean_name_inp, variance_name_inp};
    const cldnn::custom_gpu_primitive op_batch_norm(output_name,
                                                    inputs,
                                                    {writer.get_code()},
                                                    entry_point_name,
                                                    get_kernel_args(5, 1),
                                                    "",
                                                    layout,
                                                    {1});
    topology.add(op_batch_norm);
}
