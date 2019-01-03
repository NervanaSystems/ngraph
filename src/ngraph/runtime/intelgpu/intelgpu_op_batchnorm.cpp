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

// According to the documentation, input data channel is always being axis 1
// Assumed the second dimension from the left. Example {0, 1, 0, 0} or {0, 1}
// Also, input data must be at least 2D array
static const size_t channel_axis = 1;

static Shape get_channel_shape(const Shape& shape, const string& function_name)
{
    if (shape.size() < channel_axis + 1)
    {
        const string err = "intelgpu::" + function_name + "() input_shape" +
                           runtime::intelgpu::array_dims(shape) + " should be at least " +
                           to_string(channel_axis + 1) + "D.";
        throw invalid_argument(err);
    }

    return {shape.at(channel_axis)};
}

static size_t get_idx_size(const Shape& shape, size_t pos)
{
    return accumulate(shape.cbegin() + pos, shape.cend(), 1, multiplies<size_t>());
}

void runtime::intelgpu::do_create_mean(cldnn::topology& topology,
                                       const string& output_name,
                                       const element::Type& output_type,
                                       const string& input_name,
                                       const Shape& input_shape,
                                       bool backward)
{
    const Shape channel_shape = get_channel_shape(input_shape, "create_mean");
    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, channel_shape);
    const string entry_point_name = "create_mean_" + output_name;
    const size_t output_counts = shape_size<Shape>(input_shape) / input_shape.at(channel_axis);
    codegen::CodeWriter writer;

    writer << "__kernel void " << entry_point_name << "( const __global float input"
           << array_dims(input_shape) << ", __global float output" << array_dims(channel_shape)
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
            writer << "output[i" << channel_axis << "]  = sum";
            if (!backward)
            {
                writer << " / " << output_counts;
            }
            writer << ";\n";

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
                                           const element::Type& output_type,
                                           const string& input_name,
                                           const Shape& input_shape,
                                           const std::string& mean_name)
{
    const Shape channel_shape = get_channel_shape(input_shape, "create_variance");
    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, channel_shape);
    const string entry_point_name = "create_variance_" + output_name;
    const size_t output_counts = shape_size<Shape>(input_shape) / input_shape.at(channel_axis);
    codegen::CodeWriter writer;

    writer << "__kernel void " << entry_point_name << "( const __global float input"
           << array_dims(input_shape) << ", const __global float mean" << array_dims(channel_shape)
           << ", __global float output" << array_dims(channel_shape) << ")\n";

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
                                                const element::Type& output_type,
                                                double eps,
                                                const string& input_name,
                                                const Shape& input_shape,
                                                const string& gamma_name,
                                                const string& beta_name,
                                                const string& mean_name_inp,
                                                const string& variance_name_inp)
{
    const Shape channel_shape = get_channel_shape(input_shape, "batch_norm");
    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, input_shape);
    const vector<size_t> gws(input_shape.begin(), input_shape.begin() + 2);
    const string entry_point_name = "batch_norm_" + output_name;
    codegen::CodeWriter writer;

    // The kernel name and parameters
    writer << "__attribute__((reqd_work_group_size(1,1,1)))\n"
           << "__kernel void " << entry_point_name
           << "(const __global float *input0, const __global float *input1,"
           << " const __global float *input2, const __global float *input3,"
           << " const __global float *input4, __global float *output)\n";
    writer.block_begin();
    { // Main function body

        writer << "// input array dims: input0" << array_dims(input_shape);
        // Channel axis loop
        writer << "\nconst uint i" << channel_axis << " = get_global_id(" << channel_axis
               << "); /* channel_axis trip count " << input_shape.at(channel_axis) << "*/\n";

        // Invariants for the rest of the loops
        writer << "const float    gamma = input1[i" << channel_axis << "];\n"
               << "const float     beta = input2[i" << channel_axis << "];\n"
               << "const float     mean = input3[i" << channel_axis << "];\n"
               << "const float variance = input4[i" << channel_axis << "];\n"
               << "const float var_sqrt = (gamma / sqrt(variance + " << eps << "));\n";

        writer << "const uint i0 = get_global_id(0);"
               << " /* batch axis trip count " << input_shape.at(0) << "*/\n";

        // loop index invariants
        writer << "const uint idx0 = (i0 * " << get_idx_size(input_shape, 1) << ") + (i1 * "
               << get_idx_size(input_shape, 2) << ");\n";

        // SIMD loop
        writer << "for (uint i3 = 0; i3 < " << get_idx_size(input_shape, 2) << "; ++i3)\n";
        writer.block_begin();
        {
            writer << "const uint idx = idx0 + i3;\n";
            writer << "output[idx] = (input0[idx] - mean) * var_sqrt + beta;\n";
        } // Closing brackets for SIMD loop
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
                                                    gws,
                                                    {1, 1, 1});
    topology.add(op_batch_norm);
}

void runtime::intelgpu::do_create_variance_back(cldnn::topology& topology,
                                                const string& output_name,
                                                const element::Type& output_type,
                                                double eps,
                                                const string& input_name,
                                                const Shape& input_shape,
                                                const string& mean_name,
                                                const string& variance_name,
                                                const string& delta_name)
{
    const Shape channel_shape = get_channel_shape(input_shape, "create_variance_back");
    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, channel_shape);
    const string entry_point_name = "create_variance_back_" + output_name;
    codegen::CodeWriter writer;
    vector<size_t> gws;

    writer << "__kernel void " << entry_point_name << "(const __global float input"
           << array_dims(input_shape) << ", const __global float delta" << array_dims(input_shape)
           << ", const __global float mean" << array_dims(channel_shape)
           << ", const __global float variance" << array_dims(channel_shape)
           << ", __global float output" << array_dims(channel_shape) << ")\n";

    writer.block_begin();
    { // Main function body

        gws.push_back(1); //input_shape.at(0));
        // Channel axis loop
        writer << "\nconst uint i" << channel_axis << " = get_global_id(" << channel_axis
               << "); /* channel_axis trip count " << input_shape.at(channel_axis) << "*/\n";
        gws.push_back(input_shape.at(channel_axis));
        writer << "const float     mean_loc = mean[i" << channel_axis << "];\n"
               << "const float variance_loc = variance[i" << channel_axis << "];\n"
               << "const float var_sqrt = 1.0f / sqrt(variance_loc + " << eps << ");\n";
        writer << "float sum = 0.0f;\n";

        // Main loops
        writer << "for (uint i0 = 0; i0 < " << input_shape.at(0) << "; ++i0)\n";
        writer.block_begin();
        {
            writer << "for (uint i2 = 0; i2 < " << input_shape.at(2) << "; ++i2)\n";
            writer.block_begin();
            {
                writer << "for (uint i3 = 0; i3 < " << input_shape.at(3) << "; ++i3)\n";
                writer.block_begin();
                {
                    writer << "const float input_loc = input" << access_dims(input_shape) << ";\n";
                    writer << "const float delta_loc = delta" << access_dims(input_shape) << ";\n";
                    writer << "sum += (input_loc - mean_loc) * var_sqrt * delta_loc;\n";
                }
                writer.block_end();
            } // Closing brackets for Channel axis loop
            writer.block_end();
        }
        writer.block_end();
        writer << "output[i" << channel_axis << "]  = sum;\n";
    } // Main function body
    writer.block_end();

    const vector<cldnn::primitive_id>& inputs = {input_name, delta_name, mean_name, variance_name};
    const cldnn::custom_gpu_primitive op_create_variance_back(output_name,
                                                              inputs,
                                                              {writer.get_code()},
                                                              entry_point_name,
                                                              get_kernel_args(4, 1),
                                                              "",
                                                              layout,
                                                              gws);
    topology.add(op_create_variance_back);
}

void runtime::intelgpu::do_batch_norm_backprop_operation(cldnn::topology& topology,
                                                         const Shape& shape,
                                                         const element::Type& type,
                                                         const string& gamma_name,
                                                         const string& beta_name,
                                                         const string& input_name,
                                                         const string& mean_name,
                                                         const string& variance_name,
                                                         const string& delta_name,
                                                         double eps,
                                                         const string& output_name,
                                                         const string& output_gamma_name,
                                                         const string& output_beta_name)
{
    const Shape channel_shape = get_channel_shape(shape, "batch_norm_backprop");
    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(type, shape);
    const string entry_point_name = "batch_norm_backprop_" + output_name;
    const size_t r_axes_size = shape_size(shape) / shape_size(channel_shape);
    codegen::CodeWriter writer;
    vector<size_t> gws;

    writer << "__kernel void " << entry_point_name << "(const __global float input"
           << array_dims(shape) << ", const __global float delta" << array_dims(shape)
           << ", const __global float mean" << array_dims(channel_shape)
           << ", const __global float variance" << array_dims(channel_shape)
           << ", const __global float gamma" << array_dims(channel_shape)
           << ", const __global float gamma_backprop" << array_dims(channel_shape)
           << ", const __global float beta_backprop" << array_dims(channel_shape)
           << ", __global float output" << array_dims(shape) << ")\n";

    writer.block_begin();
    { // Main function body

        // Main loops
        gws = runtime::intelgpu::generate_loops(writer, shape, true);

        writer << "float stddev = sqrt(variance[i" << channel_axis << "] + " << eps << ");\n";
        writer << "float xhat = (input" << access_dims(shape) << " - mean[i" << channel_axis
               << "]) / stddev;\n";
        writer << "float norma = gamma[i" << channel_axis << "] / stddev;\n";

        writer << "output" << access_dims(shape) << " = norma * (delta" << access_dims(shape)
               << " - (xhat * gamma_backprop[i" << channel_axis << "] + beta_backprop[i"
               << channel_axis << "]) / " << r_axes_size << ");\n";

        // Closing brackets for main loops
        runtime::intelgpu::generate_loops(writer, shape, false);

    } // Main function body
    writer.block_end();

    const vector<cldnn::primitive_id>& inputs = {input_name,
                                                 delta_name,
                                                 mean_name,
                                                 variance_name,
                                                 gamma_name,
                                                 output_gamma_name,
                                                 output_beta_name};
    const cldnn::custom_gpu_primitive op_batch_norm_backprop(output_name,
                                                             inputs,
                                                             {writer.get_code()},
                                                             entry_point_name,
                                                             get_kernel_args(7, 1),
                                                             "",
                                                             layout,
                                                             gws);
    topology.add(op_batch_norm_backprop);
}
