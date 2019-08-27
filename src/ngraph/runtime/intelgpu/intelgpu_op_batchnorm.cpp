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

#include "ngraph/code_writer.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_kernels.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_custom_kernels.hpp"

#include "ngraph/op/batch_norm.hpp"

using namespace std;
using namespace ngraph;
using namespace ngraph::runtime::intelgpu;

// According to the documentation, input data channel is always being axis 1
// Assumed the second dimension from the left. Example {0, 1, 0, 0} or {0, 1}
// Also, input data must be at least 2D array
static const size_t channel_axis = 1;

static Shape get_channel_shape(const Shape& shape, const string& function_name)
{
    if (shape.size() < channel_axis + 1)
    {
        const string err = "intelgpu::" + function_name + "() input_shape" + array_dims(shape) +
                           " should be at least " + to_string(channel_axis + 1) + "D.";
        throw invalid_argument(err);
    }

    return {shape.at(channel_axis)};
}

static size_t get_idx_size(const Shape& shape, size_t pos)
{
    return accumulate(shape.cbegin() + pos, shape.cend(), 1, multiplies<size_t>());
}

// This creates mean of the input matrix by Channel axis
static CustomKernels::krnl_info do_create_mean(const string& output_name,
                                               const element::Type& output_type,
                                               const string& input_name,
                                               const Shape& input_shape,
                                               bool backward)
{
    const Shape channel_shape = get_channel_shape(input_shape, "create_mean");
    const string entry_point_name = "create_mean_" + output_name;
    const size_t output_counts = shape_size<Shape>(input_shape) / input_shape.at(channel_axis);
    const string kernel_data_type = get_opencl_type_name(output_type);
    CodeWriter writer;

    writer << "__kernel void " << entry_point_name << "( const __global " << kernel_data_type
           << " input" << array_dims(input_shape) << ", __global " << kernel_data_type << " output"
           << array_dims(channel_shape) << ")\n";

    writer.block_begin();
    { // Main function body

        // Loop for Channel axis 1
        writer << "for (uint i" << channel_axis << " = 0; i" << channel_axis << " < "
               << input_shape.at(channel_axis) << "; ++i" << channel_axis << ")\n";
        writer.block_begin();
        {
            writer << kernel_data_type << " sum = 0.0f;\n";
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

    const CustomKernelInfo op_bcast_sum(output_name,
                                        channel_shape,
                                        output_type,
                                        {input_name},
                                        {writer.get_code()},
                                        entry_point_name);
    return {op_bcast_sum};
}

// This creates variance of the input matrix by Channel axis
static CustomKernels::krnl_info do_create_variance(const string& output_name,
                                                   const element::Type& output_type,
                                                   const string& input_name,
                                                   const Shape& input_shape,
                                                   const std::string& mean_name)
{
    const Shape channel_shape = get_channel_shape(input_shape, "create_variance");
    const string entry_point_name = "create_variance_" + output_name;
    const size_t output_counts = shape_size<Shape>(input_shape) / input_shape.at(channel_axis);
    const string kernel_data_type = get_opencl_type_name(output_type);
    CodeWriter writer;

    writer << "__kernel void " << entry_point_name << "( const __global " << kernel_data_type
           << " input" << array_dims(input_shape) << ", const __global " << kernel_data_type
           << " mean" << array_dims(channel_shape) << ", __global " << kernel_data_type << " output"
           << array_dims(channel_shape) << ")\n";

    writer.block_begin();
    { // Main function body

        // Loop for Channel axis 1
        writer << "for (uint i" << channel_axis << " = 0; i" << channel_axis << " < "
               << input_shape.at(channel_axis) << "; ++i" << channel_axis << ")\n";
        writer.block_begin();
        {
            writer << kernel_data_type << " sum = 0.0f;\n";

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

            writer << kernel_data_type << " mean_diff = input" << access_dims(input_shape)
                   << " - mean[i" << channel_axis << "];\n";
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

    const CustomKernelInfo op_variance(output_name,
                                       channel_shape,
                                       output_type,
                                       {input_name, mean_name},
                                       {writer.get_code()},
                                       entry_point_name);
    return {op_variance};
}

static CustomKernels::krnl_info do_batch_norm_operation(const string& output_name,
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
    const vector<size_t> gws(input_shape.begin(), input_shape.begin() + 2);
    const string entry_point_name = "batch_norm_" + output_name;
    const string kernel_data_type = get_opencl_type_name(output_type);
    CodeWriter writer;

    // The kernel name and parameters
    writer << "__attribute__((reqd_work_group_size(1,1,1)))\n"
           << "__kernel void " << entry_point_name << "(const __global " << kernel_data_type
           << " *input0, const __global " << kernel_data_type << " *input1,"
           << " const __global " << kernel_data_type << " *input2, const __global "
           << kernel_data_type << " *input3,"
           << " const __global " << kernel_data_type << " *input4, __global " << kernel_data_type
           << " *output)\n";
    writer.block_begin();
    { // Main function body

        writer << "// input array dims: input0" << array_dims(input_shape);
        // Channel axis loop
        writer << "\nconst uint i" << channel_axis << " = get_global_id(" << channel_axis
               << "); /* channel_axis trip count " << input_shape.at(channel_axis) << "*/\n";

        // Invariants for the rest of the loops
        writer << "const " << kernel_data_type << "    gamma = input1[i" << channel_axis << "];\n"
               << "const " << kernel_data_type << "     beta = input2[i" << channel_axis << "];\n"
               << "const " << kernel_data_type << "     mean = input3[i" << channel_axis << "];\n"
               << "const " << kernel_data_type << " variance = input4[i" << channel_axis << "];\n"
               << "const " << kernel_data_type << " var_sqrt = (gamma / sqrt(variance + " << eps
               << "));\n";

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

    const vector<string>& inputs = {
        input_name, gamma_name, beta_name, mean_name_inp, variance_name_inp};
    const CustomKernelInfo op_batch_norm(output_name,
                                         input_shape,
                                         output_type,
                                         inputs,
                                         {writer.get_code()},
                                         entry_point_name,
                                         gws,
                                         {1, 1, 1});
    return {op_batch_norm};
}

// This creates variance backprop of the input matrix by Channel axis
static CustomKernels::krnl_info do_create_variance_back(const string& output_name,
                                                        const element::Type& output_type,
                                                        double eps,
                                                        const string& input_name,
                                                        const Shape& input_shape,
                                                        const string& mean_name,
                                                        const string& variance_name,
                                                        const string& delta_name)
{
    const Shape channel_shape = get_channel_shape(input_shape, "create_variance_back");
    const string entry_point_name = "create_variance_back_" + output_name;
    const string kernel_data_type = get_opencl_type_name(output_type);
    CodeWriter writer;
    vector<size_t> gws;

    writer << "__kernel void " << entry_point_name << "(const __global " << kernel_data_type
           << " input" << array_dims(input_shape) << ", const __global " << kernel_data_type
           << " delta" << array_dims(input_shape) << ", const __global " << kernel_data_type
           << " mean" << array_dims(channel_shape) << ", const __global " << kernel_data_type
           << " variance" << array_dims(channel_shape) << ", __global " << kernel_data_type
           << " output" << array_dims(channel_shape) << ")\n";

    writer.block_begin();
    { // Main function body

        gws.push_back(1); // input_shape.at(0));
        // Channel axis loop
        writer << "\nconst uint i" << channel_axis << " = get_global_id(" << channel_axis
               << "); /* channel_axis trip count " << input_shape.at(channel_axis) << "*/\n";
        gws.push_back(input_shape.at(channel_axis));
        writer << "const " << kernel_data_type << "     mean_loc = mean[i" << channel_axis << "];\n"
               << "const " << kernel_data_type << " variance_loc = variance[i" << channel_axis
               << "];\n"
               << "const " << kernel_data_type << " var_sqrt = 1.0f / sqrt(variance_loc + " << eps
               << ");\n";
        writer << kernel_data_type << " sum = 0.0f;\n";

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
                    writer << "const " << kernel_data_type << " input_loc = input"
                           << access_dims(input_shape) << ";\n";
                    writer << "const " << kernel_data_type << " delta_loc = delta"
                           << access_dims(input_shape) << ";\n";
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

    const vector<string>& inputs = {input_name, delta_name, mean_name, variance_name};
    const CustomKernelInfo op_create_variance_back(output_name,
                                                   channel_shape,
                                                   output_type,
                                                   inputs,
                                                   {writer.get_code()},
                                                   entry_point_name,
                                                   gws);
    return {op_create_variance_back};
}

// This function uses "shape" parameter as input or output Shape
// Shape of all other calculated as first axis from the left
// Example: output[ 4, 3, 2, 8 ] means out_gamma[ 3 ]
static CustomKernels::krnl_info do_batch_norm_backprop_operation(const Shape& shape,
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
    const string entry_point_name = "batch_norm_backprop_" + output_name;
    const size_t r_axes_size = shape_size(shape) / shape_size(channel_shape);
    const string kernel_data_type = get_opencl_type_name(type);
    CodeWriter writer;
    vector<size_t> gws;

    writer << "__kernel void " << entry_point_name << "(const __global " << kernel_data_type
           << " input" << array_dims(shape) << ", const __global " << kernel_data_type << " delta"
           << array_dims(shape) << ", const __global " << kernel_data_type << " mean"
           << array_dims(channel_shape) << ", const __global " << kernel_data_type << " variance"
           << array_dims(channel_shape) << ", const __global " << kernel_data_type << " gamma"
           << array_dims(channel_shape) << ", const __global " << kernel_data_type
           << " gamma_backprop" << array_dims(channel_shape) << ", const __global "
           << kernel_data_type << " beta_backprop" << array_dims(channel_shape) << ", __global "
           << kernel_data_type << " output" << array_dims(shape) << ")\n";

    writer.block_begin();
    { // Main function body

        // Main loops
        gws = generate_loops(writer, shape, true);

        writer << kernel_data_type << " stddev = sqrt(variance[i" << channel_axis << "] + " << eps
               << ");\n";
        writer << kernel_data_type << " xhat = (input" << access_dims(shape) << " - mean[i"
               << channel_axis << "]) / stddev;\n";
        writer << kernel_data_type << " norma = gamma[i" << channel_axis << "] / stddev;\n";

        writer << "output" << access_dims(shape) << " = norma * (delta" << access_dims(shape)
               << " - (xhat * gamma_backprop[i" << channel_axis << "] + beta_backprop[i"
               << channel_axis << "]) / " << r_axes_size << ");\n";

        // Closing brackets for main loops
        generate_loops(writer, shape, false);

    } // Main function body
    writer.block_end();

    const vector<string>& inputs = {input_name,
                                    delta_name,
                                    mean_name,
                                    variance_name,
                                    gamma_name,
                                    output_gamma_name,
                                    output_beta_name};
    const CustomKernelInfo op_batch_norm_backprop(
        output_name, shape, type, inputs, {writer.get_code()}, entry_point_name, gws);
    return {op_batch_norm_backprop};
}

CustomKernels::krnl_info
    CustomKernels::build_krnl(const shared_ptr<op::BatchNormInference>& op) const
{
    return do_batch_norm_operation(op->get_output_tensor_name(0),
                                   op->get_output_element_type(0),
                                   op->get_eps_value(),
                                   op->get_input_tensor_name(2),
                                   op->get_input_shape(2),
                                   op->get_input_tensor_name(0),
                                   op->get_input_tensor_name(1),
                                   op->get_input_tensor_name(3),
                                   op->get_input_tensor_name(4));
}

CustomKernels::krnl_info
    CustomKernels::build_krnl(const shared_ptr<op::BatchNormTraining>& op) const
{
    CustomKernels::krnl_info result;

    string mean_name;
    string variance_name;

    if (op->get_inputs().size() < 3 || op->get_outputs().empty())
    {
        arguments_check(op, 3, 1); // throw exception in this case
    }

    if (op->get_outputs().size() == 3)
    {
        arguments_check(op, 3, 3);

        mean_name = op->get_output_tensor_name(1);
        variance_name = op->get_output_tensor_name(2);

        CustomKernels::krnl_info mean = do_create_mean(mean_name,
                                                       op->get_output_element_type(0),
                                                       op->get_input_tensor_name(2),
                                                       op->get_input_shape(2),
                                                       false);
        result.insert(result.end(), mean.begin(), mean.end());

        CustomKernels::krnl_info variance = do_create_variance(variance_name,
                                                               op->get_output_element_type(0),
                                                               op->get_input_tensor_name(2),
                                                               op->get_input_shape(2),
                                                               mean_name);
        result.insert(result.end(), variance.begin(), variance.end());
    }

    if (op->get_outputs().size() == 1 || op->get_outputs().size() == 3)
    {
        if (mean_name.empty() || variance_name.empty())
        {
            arguments_check(op, 5, 1);

            mean_name = op->get_input_tensor_name(3);
            variance_name = op->get_input_tensor_name(4);
        }

        CustomKernels::krnl_info batch_norm =
            do_batch_norm_operation(op->get_output_tensor_name(0),
                                    op->get_output_element_type(0),
                                    op->get_eps_value(),
                                    op->get_input_tensor_name(2),
                                    op->get_input_shape(2),
                                    op->get_input_tensor_name(0),
                                    op->get_input_tensor_name(1),
                                    mean_name,
                                    variance_name);
        result.insert(result.end(), batch_norm.begin(), batch_norm.end());
    }
    else
    {
        arguments_check(op, 5, 1); // throw exception in this case
    }

    return result;
}

CustomKernels::krnl_info
    CustomKernels::build_krnl(const shared_ptr<op::BatchNormTrainingBackprop>& op) const
{
    CustomKernels::krnl_info result;

    CustomKernels::krnl_info mean = do_create_mean(op->get_output_tensor_name(2), // d_beta
                                                   op->get_output_element_type(2),
                                                   op->get_input_tensor_name(5), // delta
                                                   op->get_input_shape(5),
                                                   true);
    result.insert(result.end(), mean.begin(), mean.end());

    CustomKernels::krnl_info variance =
        do_create_variance_back(op->get_output_tensor_name(1), // d_gamma
                                op->get_output_element_type(1),
                                op->get_eps_value(),
                                op->get_input_tensor_name(2), // input
                                op->get_input_shape(2),
                                op->get_input_tensor_name(3),  // gamma
                                op->get_input_tensor_name(4),  // beta
                                op->get_input_tensor_name(5)); // delta
    result.insert(result.end(), variance.begin(), variance.end());

    CustomKernels::krnl_info batch_norm =
        do_batch_norm_backprop_operation(op->get_input_shape(2),
                                         op->get_input_element_type(2),
                                         op->get_input_tensor_name(0),
                                         op->get_input_tensor_name(1),
                                         op->get_input_tensor_name(2),
                                         op->get_input_tensor_name(3),
                                         op->get_input_tensor_name(4),
                                         op->get_input_tensor_name(5),
                                         op->get_eps_value(),
                                         op->get_output_tensor_name(0),
                                         op->get_output_tensor_name(1),
                                         op->get_output_tensor_name(2));
    result.insert(result.end(), batch_norm.begin(), batch_norm.end());

    return result;
}
