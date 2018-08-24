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

#include "ngraph/runtime/intelgpu/intelgpu_layout.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_custom_kernels.hpp"

#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

vector<cldnn_arg> runtime::intelgpu::get_kernel_args(size_t input, size_t output)
{
    vector<cldnn_arg> result;

    for (cldnn_arg_index i = 0; i < input; ++i)
    {
        result.push_back({arg_input, i});
    }

    for (cldnn_arg_index i = 0; i < output; ++i)
    {
        result.push_back({arg_output, i});
    }

    return result;
}

string runtime::intelgpu::array_dims(const Shape& dimentions, const AxisSet& axis)
{
    size_t var_idx = 0;
    string buffer;

    for (auto const& dim : dimentions)
    {
        if (axis.find(var_idx) == axis.end())
        {
            buffer += "[" + to_string(dim) + "]";
        }
        ++var_idx;
    }

    if (buffer.empty())
    { // it means scalar
        buffer = "[1]";
    }

    return buffer;
}

string
    runtime::intelgpu::access_dims(const Shape& dimentions, const AxisSet& axis, bool is_reversed)
{
    size_t var_idx = 0;
    string buffer;

    for (auto const& i : dimentions)
    {
        if (axis.find(var_idx) == axis.end())
        {
            buffer += "[i" + to_string(var_idx) + "]";
        }
        else if (is_reversed)
        {
            buffer += "[" + to_string(i) + " - i" + to_string(var_idx) + " - 1]";
        }
        ++var_idx;
    }

    if (buffer.empty())
    { // it means scalar
        buffer = "[0]";
    }

    return buffer;
}

void runtime::intelgpu::gen_func_def(codegen::CodeWriter& writer,
                                     const string& entry_point_name,
                                     const vector<string>& input_types,
                                     const vector<Shape>& input_shapes,
                                     const string& output_type,
                                     const Shape& output_shape)
{
    writer << "__kernel void " << entry_point_name << "(";

    const size_t inputs_number = input_types.size();
    for (uint i = 0; i < inputs_number; ++i)
    {
        if (i > 0)
        {
            writer << ", ";
        }
        writer << "const __global " << input_types.at(i) << " input" << i
               << array_dims(input_shapes.at(i));
    }
    writer << ", __global " << output_type << " output" << array_dims(output_shape) << ")\n";
}

vector<size_t> runtime::intelgpu::generate_loops(codegen::CodeWriter& writer,
                                                 const Shape& shape,
                                                 bool is_begin)
{
    const size_t cldnn_gws_lim = 3;
    vector<size_t> gws;
    size_t var_idx = 0;

    for (auto const& i : shape)
    {
        if (var_idx < cldnn_gws_lim)
        {
            if (is_begin)
            {
                writer << "const unsigned i" << var_idx << " = get_global_id(" << var_idx
                       << "); /*trip count " << i << "*/\n";
                gws.push_back(i);
            }
        }
        else
        {
            if (is_begin)
            {
                writer << "for (uint i" << var_idx << " = 0; i" << var_idx << " < " << i << "; ++i"
                       << var_idx << ")\n";
                writer.block_begin();
            }
            else
            {
                writer.block_end();
            }
        }
        ++var_idx;
    }

    if (gws.empty())
    {
        gws.push_back(1);
    }

    return gws;
}

static string access_dims_strided(const Shape& dimentions,
                                  const Shape& pad_below,
                                  const Shape& pad_interior,
                                  bool is_pad_interior)
{
    string buffer;
    size_t var_idx = 0;

    for (auto const& i : dimentions)
    {
        buffer += "[i" + to_string(var_idx) + " * (" + to_string(pad_interior.at(var_idx));
        if (is_pad_interior)
        {
            buffer += " + 1";
        }
        buffer += ") + " + to_string(pad_below.at(var_idx)) + "]";
        ++var_idx;
    }

    return buffer;
}

static void do_dot_operation_error(const Shape& input0_shape,
                                   const Shape& input1_shape,
                                   const Shape& output_shape)
{
    throw invalid_argument("IntelGPU Dot operation. Conbination input0_shape" +
                           runtime::intelgpu::array_dims(input0_shape) + ", input1_shape" +
                           runtime::intelgpu::array_dims(input1_shape) + ", output_shape" +
                           runtime::intelgpu::array_dims(output_shape) + " is not supported.");
}

void runtime::intelgpu::do_pad_operation(cldnn::topology& topology,
                                         const string& input_name,
                                         const Shape& input_shape,
                                         const string& scalar_name,
                                         const string& output_name,
                                         const Shape& output_shape,
                                         const element::Type& output_type,
                                         const Shape& pad_below,
                                         const Shape& pad_interior)
{
    const string entry_point_name = "op_pad_kernel_" + output_name;
    codegen::CodeWriter writer;
    vector<size_t> gws;

    // The kernel name and parameters
    runtime::intelgpu::gen_func_def(
        writer, entry_point_name, {2, "float"}, {input_shape, {1}}, "float", output_shape);

    writer.block_begin();
    {
        // Loop for Broadcast scalar over full output tensor
        gws = runtime::intelgpu::generate_loops(writer, output_shape, true);

        writer << "output" << access_dims(output_shape) << " = input1[0];\n";

        // Closing brackets for Broadcast loop
        runtime::intelgpu::generate_loops(writer, output_shape, false);

        // Loop for Copy input matrix into output matrix with padding.
        // Padding include "pad_below" and "pad_interior" according nGraph documentation
        size_t var_idx = 0;
        for (auto const& i : input_shape)
        {
            writer << "for (uint i" << var_idx << " = 0; i" << var_idx << " < " << i << "; ++i"
                   << var_idx << ")\n";
            writer.block_begin();
            ++var_idx;
        }

        writer << "output" << access_dims_strided(input_shape, pad_below, pad_interior, true)
               << " = input0" << access_dims(input_shape) << ";\n";

        // Closing brackets for main Copy loop
        for (auto const& i : input_shape)
        {
            writer.block_end();
        }

    } // End of function bracket
    writer.block_end();

    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const cldnn::custom_gpu_primitive op_pad(output_name,
                                             {input_name, scalar_name},
                                             {writer.get_code()},
                                             entry_point_name,
                                             get_kernel_args(2, 1),
                                             "",
                                             layout,
                                             gws);
    topology.add(op_pad);
}

void runtime::intelgpu::do_max_pool_backprop_operation(cldnn::topology& topology,
                                                       const string& input_name,
                                                       const Shape& input_shape,
                                                       const string& delta_name,
                                                       const Shape& delta_shape,
                                                       const string& output_name,
                                                       const Shape& output_shape,
                                                       const element::Type& output_type,
                                                       const Shape& win_shape,
                                                       const Shape& win_stride,
                                                       const Shape& pad_below)
{
    const string entry_point_name = "op_max_pool_backprop_" + output_name;
    const Shape delta_data(delta_shape.cbegin() + 2, delta_shape.cend());
    const Shape output_data(output_shape.cbegin() + 2, output_shape.cend());
    codegen::CodeWriter writer;
    vector<size_t> gws;

    // The kernel name and parameters
    runtime::intelgpu::gen_func_def(
        writer, entry_point_name, {2, "float"}, {input_shape, delta_shape}, "float", output_shape);

    writer.block_begin();
    {
        // Main loop over delta input array.
        writer << "const uint i0 = get_global_id(0);";
        gws.push_back(delta_shape.at(0));
        writer << "// for (uint i0 = 0; i0 < " << delta_shape.at(0) << "; ++i0)\n";
        writer.block_begin();
        {
            writer << "const uint i1 = get_global_id(1);";
            gws.push_back(delta_shape.at(1));
            writer << "// for (uint i1 = 0; i1 < " << delta_shape.at(1) << "; ++i1)\n";
            writer.block_begin();
            {
                // Initialization output
                size_t var_idx = 0;
                for (auto const& i : output_data)
                {
                    writer << "for (uint j" << var_idx << " = 0; j" << var_idx << " < " << i
                           << "; ++j" << var_idx << ")\n";
                    writer.block_begin();
                    ++var_idx;
                }

                writer << "output[i0][i1]";
                // Additional dimentions for output
                for (size_t i = 0; i < output_data.size(); ++i)
                {
                    writer << "[j" << i << "]";
                }
                writer << " = 0.0f;\n";

                // Closing brackets for Initialization loop
                for (auto const& i : output_data)
                {
                    writer.block_end();
                }
                // End of output initialization

                // Loops over other output dimensions
                var_idx = 2;
                for (auto const& i : delta_data)
                {
                    writer << "for (uint i" << var_idx << " = 0; i" << var_idx << " < " << i
                           << "; ++i" << var_idx << ")\n";
                    writer.block_begin();
                    ++var_idx;
                }

                // Create variables to save coordinates
                for (size_t i = 0; i < delta_data.size(); ++i)
                {
                    writer << "uint save_i" << i + 2 << " = 0;\n";
                }
                writer << "float max_elem = FLT_MIN;\n"
                       << "uint elem_exists = 0;\n";

                // Loop over window shape
                var_idx = 0;
                for (auto const& i : win_shape)
                {
                    writer << "for (uint w" << var_idx << " = 0; w" << var_idx << " < " << i
                           << "; ++w" << var_idx << ")\n";
                    writer.block_begin();
                    writer << "const uint win_idx" << var_idx << " = (i" << var_idx + 2 << " * "
                           << win_stride.at(var_idx) << " /*win_stride*/)"
                           << " + w" << var_idx << " - " << pad_below.at(var_idx)
                           << " /*pad_below*/;\n";
                    ++var_idx;
                }

                // input coordinate condition
                writer << "if (";
                // Generate input coordinate condition
                for (size_t i = 0; i < win_shape.size(); ++i)
                {
                    if (i)
                    {
                        writer << " && ";
                    }
                    writer << "(win_idx" << i << " < " << input_shape.at(i + 2) << ")";
                }
                writer << ")\n";
                writer.block_begin();
                {
                    writer << "const float max_local = input0[i0][i1]";
                    // additional dimensions for input
                    for (size_t i = 0; i < win_shape.size(); ++i)
                    {
                        writer << "[win_idx" << i << "]";
                    }
                    writer << ";\n";

                    // find maximum condition
                    writer << "if (max_local > max_elem)\n";
                    writer.block_begin();
                    {
                        writer << "max_elem = max_local;\n"
                               << "elem_exists = 1;\n";

                        // Save coordinates
                        for (size_t i = 0; i < delta_data.size(); ++i)
                        {
                            writer << "save_i" << i + 2 << " = win_idx" << i << ";\n";
                        }
                    } // End of find maximum condition
                    writer.block_end();

                } // End of input coordinate condition
                writer.block_end();
                // Closing brackets for window shape loop
                for (auto const& i : win_shape)
                {
                    writer.block_end();
                }

                // Elem_exists condition
                writer << "if (elem_exists)\n";
                writer.block_begin();
                {
                    writer << "output[i0][i1]";
                    // Additional dimentions for output
                    for (size_t i = 0; i < delta_data.size(); ++i)
                    {
                        writer << "[save_i" << i + 2 << "]";
                    }
                    writer << " += input1" << access_dims(delta_shape) << ";\n";
                } // End of elem_exists condition
                writer.block_end();
                // Closing brackets for delta loop
                for (auto const& i : delta_data)
                {
                    writer.block_end();
                }
            } // End of loop over i1
            writer.block_end();
        } // End of loop over i0
        writer.block_end();

    } // End of function bracket
    writer.block_end();

    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const cldnn::custom_gpu_primitive op_max_pool_backprop(output_name,
                                                           {input_name, delta_name},
                                                           {writer.get_code()},
                                                           entry_point_name,
                                                           get_kernel_args(2, 1),
                                                           "",
                                                           layout,
                                                           gws);
    topology.add(op_max_pool_backprop);
}

static void do_1d_scalar_mul(codegen::CodeWriter& writer,
                             string& entry_point_name,
                             const Shape& input0_shape,
                             const Shape& input1_shape)
{
    const size_t input0_count = input0_shape.empty() ? 0 : shape_size<Shape>(input0_shape);
    const size_t input1_count = input1_shape.empty() ? 0 : shape_size<Shape>(input1_shape);
    const size_t output_count = max(input0_count, input1_count);
    entry_point_name += "_do_1d_scalar_mul";

    writer << "__kernel void " << entry_point_name << "(const __global float* input0"
           << ", const __global float* input1, __global float* output)\n";
    writer.block_begin();
    {
        writer << "for (uint i1 = 0; i1 < " << output_count << "; ++i1)\n";
        writer.block_begin();
        {
            writer << "output[i1] = input0[" << (input0_count > 0 ? "i1" : "0") << "] * input1["
                   << (input1_count > 0 ? "i1" : "0") << "];\n";
        }
        writer.block_end();
    }
    writer.block_end();
}

static vector<size_t> do_2d_2d_mul(codegen::CodeWriter& writer,
                                   string& entry_point_name,
                                   const Shape& input0_shape,
                                   const Shape& input1_shape,
                                   const Shape& output_shape)
{
    entry_point_name += "_do_2d_2d_mul";
    vector<size_t> gws;

    runtime::intelgpu::gen_func_def(writer,
                                    entry_point_name,
                                    {2, "float"},
                                    {input0_shape, input1_shape},
                                    "float",
                                    output_shape);

    writer.block_begin();
    {
        // Main loops
        gws = runtime::intelgpu::generate_loops(writer, output_shape, true);

        // Inner loop
        writer << "float sum = 0.0f;\n";
        writer << "for (uint i2 = 0; i2 < " << input0_shape.at(1) << "; ++i2)\n";
        writer.block_begin();
        {
            writer << "sum += input0[i0][i2] * input1[i2][i1];\n";
        }
        writer.block_end();
        writer << "output[i0][i1] = sum;\n";

        // Closing brackets for main loops
        runtime::intelgpu::generate_loops(writer, output_shape, false);
    }
    writer.block_end();

    return gws;
}

static vector<size_t> do_3d_3d_mul(codegen::CodeWriter& writer,
                                   string& entry_point_name,
                                   const Shape& input0_shape,
                                   const Shape& input1_shape,
                                   const Shape& output_shape)
{
    entry_point_name += "_do_3d_3d_mul";
    vector<size_t> gws;

    runtime::intelgpu::gen_func_def(writer,
                                    entry_point_name,
                                    {2, "float"},
                                    {input0_shape, input1_shape},
                                    "float",
                                    output_shape);

    writer.block_begin();
    {
        // Main loops
        gws = runtime::intelgpu::generate_loops(writer, output_shape, true);

        // Inner loop
        writer << "float sum = 0.0f;\n";
        writer << "for (uint i4 = 0; i4 < " << input0_shape.back() << "; ++i4)\n";
        writer.block_begin();
        {
            writer << "sum += input0[i0][i1][i4] * input1[i4][i2][i3];\n";
        }
        writer.block_end();
        writer << "output[i0][i1][i2][i3] = sum;\n";

        // Closing brackets for main loops
        runtime::intelgpu::generate_loops(writer, output_shape, false);
    }
    writer.block_end();

    return gws;
}

static vector<size_t> do_3d_2d_mul(codegen::CodeWriter& writer,
                                   string& entry_point_name,
                                   const Shape& input0_shape,
                                   const Shape& input1_shape,
                                   const Shape& output_shape)
{
    entry_point_name += "_do_3d_2d_mul";
    vector<size_t> gws;

    runtime::intelgpu::gen_func_def(writer,
                                    entry_point_name,
                                    {2, "float"},
                                    {input0_shape, input1_shape},
                                    "float",
                                    output_shape);

    writer.block_begin();
    {
        // Main loops
        gws = runtime::intelgpu::generate_loops(writer, output_shape, true);

        // Inner loop
        writer << "float sum = 0.0f;\n";
        writer << "for (uint i3 = 0; i3 < " << input0_shape.back() << "; ++i3)\n";
        writer.block_begin();
        {
            writer << "sum += input0[i0][i1][i3] * input1[i3][i2];\n";
        }
        writer.block_end();
        writer << "output[i0][i1][i2] = sum;\n";

        // Closing brackets for main loops
        runtime::intelgpu::generate_loops(writer, output_shape, false);
    }
    writer.block_end();

    return gws;
}

static vector<size_t> do_2d_1d_mul(codegen::CodeWriter& writer,
                                   string& entry_point_name,
                                   const Shape& input0_shape,
                                   const Shape& input1_shape,
                                   const Shape& output_shape)
{
    entry_point_name += "_do_2d_1d_mul";
    vector<size_t> gws;

    runtime::intelgpu::gen_func_def(writer,
                                    entry_point_name,
                                    {2, "float"},
                                    {input0_shape, input1_shape},
                                    "float",
                                    output_shape);

    writer.block_begin();
    {
        // Main loops
        gws = runtime::intelgpu::generate_loops(writer, output_shape, true);

        writer << "float sum = 0.0f;\n";
        // Inner loop
        writer << "for (uint i1 = 0; i1 < " << input0_shape.at(1) << "; ++i1)\n";
        writer.block_begin();
        {
            writer << "sum += input0[i0][i1] * input1[i1];\n";
        }
        writer.block_end();
        writer << "output[i0] = sum;\n";

        // Closing brackets for main loops
        runtime::intelgpu::generate_loops(writer, output_shape, false);
    }
    writer.block_end();

    return gws;
}

static void do_scalar_scalar_mul(codegen::CodeWriter& writer, string& entry_point_name)
{
    entry_point_name += "_scalar_scalar_mul";

    runtime::intelgpu::gen_func_def(
        writer, entry_point_name, {2, "float"}, {{1}, {1}}, "float", {1});

    writer.block_begin();
    {
        writer << "output[0] = input0[0] * input1[0];\n";
    }
    writer.block_end();
}

static void do_1d_1d_mul(codegen::CodeWriter& writer, string& entry_point_name, const Shape& shape)
{
    if (shape.size() > 1)
    {
        throw invalid_argument("do_1d_1d_mul: Shape" + runtime::intelgpu::array_dims(shape) +
                               " must be 1D");
    }

    entry_point_name += "_do_1d_1d_mul";

    runtime::intelgpu::gen_func_def(
        writer, entry_point_name, {2, "float"}, {2, shape}, "float", {1});

    writer.block_begin();
    {
        writer << "float sum = 0.0f;\n"
               << "for (uint i = 0; i < " << shape.front() << "; ++i)\n";
        writer.block_begin();
        {
            writer << "sum += input0[i] * input1[i];\n";
        }
        writer.block_end();
        writer << "output[0] = sum;\n";
    }
    writer.block_end();
}

void runtime::intelgpu::do_dot_operation(cldnn::topology& topology,
                                         const string& input0_name,
                                         const Shape& input0_shape,
                                         const string& input1_name,
                                         const Shape& input1_shape,
                                         const string& output_name,
                                         const Shape& output_shape,
                                         const element::Type& output_type)
{
    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    string entry_point_name = "dot_" + output_name;
    codegen::CodeWriter writer;
    vector<size_t> gws = {1};

    const bool is_input0_scalar = input0_shape.empty();
    const bool is_input1_scalar = input1_shape.empty();
    const bool is_output_scalar = output_shape.empty();

    if (is_input0_scalar && is_input1_scalar && is_output_scalar)
    {
        do_scalar_scalar_mul(writer, entry_point_name);
    }
    else if (((is_input0_scalar && !is_input1_scalar) || (!is_input0_scalar && is_input1_scalar)) &&
             !is_output_scalar)
    {
        do_1d_scalar_mul(writer, entry_point_name, input0_shape, input1_shape);
    }
    else if (!is_input0_scalar && !is_input1_scalar && is_output_scalar)
    {
        do_1d_1d_mul(writer, entry_point_name, input1_shape);
    }
    else if (!is_input0_scalar && !is_input1_scalar && !is_output_scalar)
    {
        if (input0_shape.size() == 2 && input1_shape.size() == 1)
        {
            gws = do_2d_1d_mul(writer, entry_point_name, input0_shape, input1_shape, output_shape);
        }
        else if (input0_shape.size() == 2 && input1_shape.size() == 2)
        {
            gws = do_2d_2d_mul(writer, entry_point_name, input0_shape, input1_shape, output_shape);
        }
        else if (input0_shape.size() == 3 && input1_shape.size() == 3)
        {
            gws = do_3d_3d_mul(writer, entry_point_name, input0_shape, input1_shape, output_shape);
        }
        else if (input0_shape.size() == 3 && input1_shape.size() == 2)
        {
            gws = do_3d_2d_mul(writer, entry_point_name, input0_shape, input1_shape, output_shape);
        }
        else
        {
            do_dot_operation_error(input0_shape, input1_shape, output_shape);
        }
    }
    else
    {
        do_dot_operation_error(input0_shape, input1_shape, output_shape);
    }

    const cldnn::custom_gpu_primitive op_dot(output_name,
                                             {input0_name, input1_name},
                                             {writer.get_code()},
                                             entry_point_name,
                                             get_kernel_args(2, 1),
                                             "",
                                             layout,
                                             gws);
    topology.add(op_dot);
}

void runtime::intelgpu::do_slice_operation(cldnn::topology& topology,
                                           const string& input_name,
                                           const Shape& input_shape,
                                           const string& output_name,
                                           const Shape& output_shape,
                                           const element::Type& output_type,
                                           const Coordinate& lower_bounds,
                                           const Coordinate& uppper_bounds,
                                           const Strides& strides)
{
    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const string entry_point_name = "slice_" + output_name;
    codegen::CodeWriter writer;
    vector<size_t> gws;

    runtime::intelgpu::gen_func_def(
        writer, entry_point_name, {"float"}, {input_shape}, "float", output_shape);

    writer.block_begin();
    {
        // Main loops
        gws = runtime::intelgpu::generate_loops(writer, output_shape, true);

        writer << "output" << access_dims(output_shape) << " = input0"
               << access_dims_strided(input_shape, lower_bounds, strides, false) << ";\n";

        // Closing brackets for main loops
        runtime::intelgpu::generate_loops(writer, output_shape, false);
    }
    writer.block_end();

    const cldnn::custom_gpu_primitive op_slice(output_name,
                                               {input_name},
                                               {writer.get_code()},
                                               entry_point_name,
                                               get_kernel_args(1, 1),
                                               "",
                                               layout,
                                               gws);
    topology.add(op_slice);
}

void runtime::intelgpu::do_select_operation(cldnn::topology& topology,
                                            const string& input0_name,
                                            const Shape& input0_shape,
                                            const string& input1_name,
                                            const Shape& input1_shape,
                                            const string& input2_name,
                                            const Shape& input2_shape,
                                            const string& output_name,
                                            const Shape& output_shape,
                                            const element::Type& output_type)
{
    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const string entry_point_name = "select_" + output_name;
    codegen::CodeWriter writer;
    vector<size_t> gws;

    runtime::intelgpu::gen_func_def(writer,
                                    entry_point_name,
                                    {"char", "float", "float"},
                                    {input0_shape, input1_shape, input2_shape},
                                    "float",
                                    output_shape);

    writer.block_begin();
    {
        // Main loops
        gws = runtime::intelgpu::generate_loops(writer, output_shape, true);

        writer << "output" << access_dims(output_shape) << " = input0" << access_dims(input0_shape)
               << " ? input1" << access_dims(input1_shape) << " : input2"
               << access_dims(input2_shape) << ";\n";

        // Closing brackets for main loops
        runtime::intelgpu::generate_loops(writer, output_shape, false);
    }
    writer.block_end();

    const cldnn::custom_gpu_primitive op_select(output_name,
                                                {input0_name, input1_name, input2_name},
                                                {writer.get_code()},
                                                entry_point_name,
                                                get_kernel_args(3, 1),
                                                "",
                                                layout,
                                                gws);
    topology.add(op_select);
}

void runtime::intelgpu::do_logic_kernel(cldnn::topology& topology,
                                        const string& input0_name,
                                        const Shape& input0_shape,
                                        const string& input0_type,
                                        const string& input1_name,
                                        const Shape& input1_shape,
                                        const string& input1_type,
                                        const string& output_name,
                                        const Shape& output_shape,
                                        const element::Type& output_type,
                                        const string& operation)
{
    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const string entry_point_name = "logic_" + output_name;
    codegen::CodeWriter writer;
    vector<size_t> gws;

    runtime::intelgpu::gen_func_def(writer,
                                    entry_point_name,
                                    {2, input0_type},
                                    {input0_shape, input1_shape},
                                    "char",
                                    output_shape);

    writer.block_begin();
    {
        // Main loops
        gws = runtime::intelgpu::generate_loops(writer, output_shape, true);

        writer << "output" << access_dims(output_shape) << " = input0" << access_dims(input0_shape)
               << operation << "input1" << access_dims(input1_shape) << " ? 1 : 0;\n";

        // Closing brackets for main loops
        runtime::intelgpu::generate_loops(writer, output_shape, false);
    }
    writer.block_end();

    const cldnn::custom_gpu_primitive op_logical(output_name,
                                                 {input0_name, input1_name},
                                                 {writer.get_code()},
                                                 entry_point_name,
                                                 get_kernel_args(2, 1),
                                                 "",
                                                 layout,
                                                 gws);
    topology.add(op_logical);
}

void runtime::intelgpu::do_reverse_operation(cldnn::topology& topology,
                                             const string& input_name,
                                             const Shape& input_shape,
                                             const string& output_name,
                                             const Shape& output_shape,
                                             const element::Type& output_type,
                                             const AxisSet& reversed_axes)
{
    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const string entry_point_name = "reverse_" + output_name;
    codegen::CodeWriter writer;
    vector<size_t> gws;

    runtime::intelgpu::gen_func_def(
        writer, entry_point_name, {"float"}, {input_shape}, "float", output_shape);

    writer.block_begin();
    {
        gws = generate_loops(writer, output_shape, true);

        writer << "output" << access_dims(output_shape) << " = input0"
               << access_dims(output_shape, reversed_axes, true) << ";\n";

        generate_loops(writer, output_shape, false);
    }
    writer.block_end();

    const cldnn::custom_gpu_primitive op_reverse(output_name,
                                                 {input_name},
                                                 {writer.get_code()},
                                                 entry_point_name,
                                                 get_kernel_args(1, 1),
                                                 "",
                                                 layout,
                                                 gws);
    topology.add(op_reverse);
}

void runtime::intelgpu::do_not_operation(cldnn::topology& topology,
                                         const string& input_name,
                                         const Shape& input_shape,
                                         const string& output_name,
                                         const Shape& output_shape,
                                         const element::Type& output_type)
{
    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const string entry_point_name = "logic_" + output_name;
    codegen::CodeWriter writer;
    vector<size_t> gws;

    runtime::intelgpu::gen_func_def(
        writer, entry_point_name, {"char"}, {input_shape}, "char", output_shape);

    writer.block_begin();
    {
        gws = runtime::intelgpu::generate_loops(writer, output_shape, true);

        writer << "output" << access_dims(output_shape) << " = !input0" << access_dims(input_shape)
               << ";\n";

        runtime::intelgpu::generate_loops(writer, output_shape, false);
    }
    writer.block_end();

    const cldnn::custom_gpu_primitive op_not(output_name,
                                             {input_name},
                                             {writer.get_code()},
                                             entry_point_name,
                                             get_kernel_args(1, 1),
                                             "",
                                             layout,
                                             gws);
    topology.add(op_not);
}

void runtime::intelgpu::do_one_hot_operation(cldnn::topology& topology,
                                             const std::string& input_name,
                                             const Shape& input_shape,
                                             const element::Type& input_type,
                                             const std::string& output_name,
                                             const Shape& output_shape,
                                             const element::Type& output_type,
                                             const size_t one_hot_axis)
{
    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const string entry_point_name = "one_hot_" + output_name;
    codegen::CodeWriter writer;
    vector<size_t> gws;

    runtime::intelgpu::gen_func_def(writer,
                                    entry_point_name,
                                    {input_type.c_type_string()},
                                    {input_shape},
                                    output_type.c_type_string(),
                                    output_shape);

    writer.block_begin();
    {
        writer << "for (uint i = 0; i < " << output_shape.at(one_hot_axis) << "; ++i)\n";
        writer.block_begin();
        {
            gws = runtime::intelgpu::generate_loops(writer, input_shape, true);

            size_t current_input = 0;
            string buffer;
            const size_t output_shape_size = output_shape.size();
            for (uint j = 0; j < output_shape_size; j++)
            {
                if (j == one_hot_axis)
                {
                    buffer += "[i]";
                }
                else
                {
                    buffer += "[i" + to_string(current_input) + "]";
                    ++current_input;
                }
            }

            writer << "output" << buffer << " = input0" << access_dims(input_shape)
                   << " == i ? 1 : 0;\n";

            runtime::intelgpu::generate_loops(writer, input_shape, false);
        }
        writer.block_end();
    }
    writer.block_end();

    const cldnn::custom_gpu_primitive op_one_hot(output_name,
                                                 {input_name},
                                                 {writer.get_code()},
                                                 entry_point_name,
                                                 get_kernel_args(1, 1),
                                                 "",
                                                 layout,
                                                 gws);
    topology.add(op_one_hot);
}

void runtime::intelgpu::do_convert_operation(cldnn::topology& topology,
                                             const string& input_name,
                                             const Shape& input_shape,
                                             const element::Type& input_type,
                                             const string& output_name,
                                             const Shape& output_shape,
                                             const element::Type& output_type)
{
    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const string entry_point_name = "convert_" + output_name;
    const string& input_type_name = input_type.c_type_string();
    const string& output_type_name = output_type.c_type_string();
    codegen::CodeWriter writer;
    vector<size_t> gws;

    runtime::intelgpu::gen_func_def(
        writer, entry_point_name, {input_type_name}, {input_shape}, output_type_name, output_shape);

    writer.block_begin();
    {
        gws = generate_loops(writer, output_shape, true);

        writer << "output" << access_dims(output_shape) << " = convert_" << output_type_name
               << "(input0" << access_dims(output_shape) << ");\n";

        generate_loops(writer, output_shape, false);
    }
    writer.block_end();

    const cldnn::custom_gpu_primitive op_convert(output_name,
                                                 {input_name},
                                                 {writer.get_code()},
                                                 entry_point_name,
                                                 get_kernel_args(1, 1),
                                                 "",
                                                 layout,
                                                 gws);
    topology.add(op_convert);
}
