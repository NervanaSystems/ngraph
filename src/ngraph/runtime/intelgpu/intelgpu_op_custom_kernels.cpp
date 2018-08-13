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

string runtime::intelgpu::array_dims(const Shape& dimentions)
{
    string buffer;

    for (auto const& dim : dimentions)
    {
        buffer += "[" + to_string(dim) + "]";
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
                writer << "const unsigned i" << var_idx << " = get_global_id(" << var_idx << ");\n";
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

static void do_dot_operation_error(const Shape& shapeA, const Shape& shapeB, const Shape& shapeZ)
{
    throw invalid_argument("IntelGPU Dot operation. Conbination ShapeA" +
                           runtime::intelgpu::array_dims(shapeA) + ", ShapeB" +
                           runtime::intelgpu::array_dims(shapeB) + ", ShapeOutput" +
                           runtime::intelgpu::array_dims(shapeZ) + " is not supported.");
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
    writer << "__kernel void " << entry_point_name << "(const __global float input"
           << array_dims(input_shape) << ", const __global float scalar[1], __global float output"
           << array_dims(output_shape) << ")\n";

    writer.block_begin();
    {
        // Loop for Broadcast scalar over full output tensor
        gws = runtime::intelgpu::generate_loops(writer, output_shape, true);

        writer << "output" << access_dims(output_shape) << " = scalar[0];\n";

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
               << " = input" << access_dims(input_shape) << ";\n";

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

static void do_1d_scalar_mul(codegen::CodeWriter& writer,
                             string& kernel_name,
                             const Shape& shapeA,
                             const Shape& shapeB)
{
    const size_t countA = shapeA.empty() ? 0 : shape_size<Shape>(shapeA);
    const size_t countB = shapeB.empty() ? 0 : shape_size<Shape>(shapeB);
    const size_t countZ = max(countA, countB);
    kernel_name += "_do_1d_scalar_mul";

    writer << "__kernel void " << kernel_name << "(const __global float* inputA"
           << ", const __global float* inputB, __global float* output)\n";
    writer.block_begin();
    {
        writer << "for (uint i1 = 0; i1 < " << countZ << "; ++i1)\n";
        writer.block_begin();
        {
            writer << "output[i1] = inputA[" << (countA > 0 ? "i1" : "0") << "] * inputB["
                   << (countB > 0 ? "i1" : "0") << "];\n";
        }
        writer.block_end();
    }
    writer.block_end();
}

static vector<size_t> do_2d_2d_mul(codegen::CodeWriter& writer,
                                   string& kernel_name,
                                   const Shape& shapeA,
                                   const Shape& shapeB,
                                   const Shape& shapeZ)
{
    const size_t colrow = shapeA.at(1);
    kernel_name += "_do_2d_2d_mul";
    vector<size_t> gws;

    writer << "__kernel void " << kernel_name << "(const __global float inputA"
           << runtime::intelgpu::array_dims(shapeA) << ", const __global float inputB"
           << runtime::intelgpu::array_dims(shapeB) << ", __global float output"
           << runtime::intelgpu::array_dims(shapeZ) << ")\n";
    writer.block_begin();
    {
        size_t var_idx = 0;
        // Main loops
        gws = runtime::intelgpu::generate_loops(writer, shapeZ, true);

        // Inner loop
        writer << "float sum = 0.0f;\n";
        writer << "for (uint i2 = 0; i2 < " << colrow << "; ++i2)\n";
        writer.block_begin();
        {
            writer << "sum += inputA[i0][i2] * inputB[i2][i1];\n";
        }
        writer.block_end();
        writer << "output[i0][i1] = sum;\n";

        // Closing brackets for main loops
        runtime::intelgpu::generate_loops(writer, shapeZ, false);
    }
    writer.block_end();

    return gws;
}

static vector<size_t> do_3d_3d_mul(codegen::CodeWriter& writer,
                                   string& kernel_name,
                                   const Shape& shapeA,
                                   const Shape& shapeB,
                                   const Shape& shapeZ)
{
    const size_t colrow = shapeA.back();
    kernel_name += "_do_3d_3d_mul";
    vector<size_t> gws;

    writer << "__kernel void " << kernel_name << "(const __global float inputA"
           << runtime::intelgpu::array_dims(shapeA) << ", const __global float inputB"
           << runtime::intelgpu::array_dims(shapeB) << ", __global float output"
           << runtime::intelgpu::array_dims(shapeZ) << ")\n";
    writer.block_begin();
    {
        size_t var_idx = 0;
        // Main loops
        gws = runtime::intelgpu::generate_loops(writer, shapeZ, true);

        // Inner loop
        writer << "float sum = 0.0f;\n";
        writer << "for (uint i4 = 0; i4 < " << colrow << "; ++i4)\n";
        writer.block_begin();
        {
            writer << "sum += inputA[i0][i1][i4] * inputB[i4][i2][i3];\n";
        }
        writer.block_end();
        writer << "output[i0][i1][i2][i3] = sum;\n";

        // Closing brackets for main loops
        runtime::intelgpu::generate_loops(writer, shapeZ, false);
    }
    writer.block_end();

    return gws;
}

static vector<size_t> do_3d_2d_mul(codegen::CodeWriter& writer,
                                   string& kernel_name,
                                   const Shape& shapeA,
                                   const Shape& shapeB,
                                   const Shape& shapeZ)
{
    const size_t colrow = shapeA.back();
    kernel_name += "_do_3d_2d_mul";
    vector<size_t> gws;

    writer << "__kernel void " << kernel_name << "(const __global float inputA"
           << runtime::intelgpu::array_dims(shapeA) << ", const __global float inputB"
           << runtime::intelgpu::array_dims(shapeB) << ", __global float output"
           << runtime::intelgpu::array_dims(shapeZ) << ")\n";
    writer.block_begin();
    {
        size_t var_idx = 0;
        // Main loops
        gws = runtime::intelgpu::generate_loops(writer, shapeZ, true);

        // Inner loop
        writer << "float sum = 0.0f;\n";
        writer << "for (uint i3 = 0; i3 < " << colrow << "; ++i3)\n";
        writer.block_begin();
        {
            writer << "sum += inputA[i0][i1][i3] * inputB[i3][i2];\n";
        }
        writer.block_end();
        writer << "output[i0][i1][i2] = sum;\n";

        // Closing brackets for main loops
        runtime::intelgpu::generate_loops(writer, shapeZ, false);
    }
    writer.block_end();

    return gws;
}

static vector<size_t> do_2d_1d_mul(codegen::CodeWriter& writer,
                                   string& kernel_name,
                                   const Shape& shapeA,
                                   const Shape& shapeB,
                                   const Shape& shapeZ)
{
    const size_t colrow = shapeA.at(1);
    kernel_name += "_do_2d_1d_mul";
    vector<size_t> gws;

    writer << "__kernel void " << kernel_name << "(const __global float inputA"
           << runtime::intelgpu::array_dims(shapeA) << ", const __global float inputB"
           << runtime::intelgpu::array_dims(shapeB) << ", __global float output"
           << runtime::intelgpu::array_dims(shapeZ) << ")\n";
    writer.block_begin();
    {
        // Main loops
        gws = runtime::intelgpu::generate_loops(writer, shapeZ, true);

        writer << "float sum = 0.0f;\n";
        // Inner loop
        writer << "for (uint i1 = 0; i1 < " << colrow << "; ++i1)\n";
        writer.block_begin();
        {
            writer << "sum += inputA[i0][i1] * inputB[i1];\n";
        }
        writer.block_end();
        writer << "output[i0] = sum;\n";

        // Closing brackets for main loops
        runtime::intelgpu::generate_loops(writer, shapeZ, false);
    }
    writer.block_end();

    return gws;
}

static void do_scalar_scalar_mul(codegen::CodeWriter& writer, string& kernel_name)
{
    kernel_name += "_scalar_scalar_mul";

    writer << "__kernel void " << kernel_name << "(const __global float inputA[1]"
           << ", const __global float inputB[1], __global float output[1])\n";
    writer.block_begin();
    {
        writer << "output[0] = inputA[0] * inputB[0];\n";
    }
    writer.block_end();
}

static void do_1d_1d_mul(codegen::CodeWriter& writer, string& kernel_name, const Shape& shape)
{
    if (shape.size() > 1)
    {
        throw invalid_argument("do_1d_1d_mul: Shape" + runtime::intelgpu::array_dims(shape) +
                               " must be 1D");
    }

    const size_t& size = shape.front();
    kernel_name += "_do_1d_1d_mul";

    writer << "__kernel void " << kernel_name << "(const __global float inputA"
           << runtime::intelgpu::array_dims(shape) << ", const __global float inputB"
           << runtime::intelgpu::array_dims(shape) << ", __global float output[1])\n";
    writer.block_begin();
    {
        writer << "float sum = 0.0f;\n"
               << "for (uint i = 0; i < " << size << "; ++i)\n";
        writer.block_begin();
        {
            writer << "sum += inputA[i] * inputB[i];\n";
        }
        writer.block_end();
        writer << "output[0] = sum;\n";
    }
    writer.block_end();
}

void runtime::intelgpu::do_dot_operation(cldnn::topology& topology,
                                         const string& inputA_name,
                                         const Shape& inputA_shape,
                                         const string& inputB_name,
                                         const Shape& inputB_shape,
                                         const string& output_name,
                                         const Shape& output_shape,
                                         const element::Type& output_type)
{
    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    string entry_point_name = "dot_" + output_name;
    codegen::CodeWriter writer;
    vector<size_t> gws = {1};

    const bool A_is_scalar = inputA_shape.empty();
    const bool B_is_scalar = inputB_shape.empty();
    const bool Z_is_scalar = output_shape.empty();

    if (A_is_scalar && B_is_scalar && Z_is_scalar)
    {
        do_scalar_scalar_mul(writer, entry_point_name);
    }
    else if (((A_is_scalar && !B_is_scalar) || (!A_is_scalar && B_is_scalar)) && !Z_is_scalar)
    {
        do_1d_scalar_mul(writer, entry_point_name, inputA_shape, inputB_shape);
    }
    else if (!A_is_scalar && !B_is_scalar && Z_is_scalar)
    {
        do_1d_1d_mul(writer, entry_point_name, inputB_shape);
    }
    else if (!A_is_scalar && !B_is_scalar && !Z_is_scalar)
    {
        if (inputA_shape.size() == 2 && inputB_shape.size() == 1)
        {
            gws = do_2d_1d_mul(writer, entry_point_name, inputA_shape, inputB_shape, output_shape);
        }
        else if (inputA_shape.size() == 2 && inputB_shape.size() == 2)
        {
            gws = do_2d_2d_mul(writer, entry_point_name, inputA_shape, inputB_shape, output_shape);
        }
        else if (inputA_shape.size() == 3 && inputB_shape.size() == 3)
        {
            gws = do_3d_3d_mul(writer, entry_point_name, inputA_shape, inputB_shape, output_shape);
        }
        else if (inputA_shape.size() == 3 && inputB_shape.size() == 2)
        {
            gws = do_3d_2d_mul(writer, entry_point_name, inputA_shape, inputB_shape, output_shape);
        }
        else
        {
            do_dot_operation_error(inputA_shape, inputB_shape, output_shape);
        }
    }
    else
    {
        do_dot_operation_error(inputA_shape, inputB_shape, output_shape);
    }

    const cldnn::custom_gpu_primitive op_dot(output_name,
                                             {inputA_name, inputB_name},
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

    writer << "__kernel void " << entry_point_name << "(const __global float input"
           << array_dims(input_shape) << ", __global float output" << array_dims(output_shape)
           << ")\n";
    writer.block_begin();
    {
        size_t var_idx = 0;
        // Main loops
        for (auto const& i : output_shape)
        {
            writer << "for (uint i" << var_idx << " = 0; i" << var_idx << " < " << i << "; ++i"
                   << var_idx << ")\n";
            writer.block_begin();
            ++var_idx;
        }

        writer << "output" << access_dims(output_shape) << " = input"
               << access_dims_strided(input_shape, lower_bounds, strides, false) << ";\n";

        // Closing brackets for main loops
        for (auto const& i : output_shape)
        {
            writer.block_end();
        }
    }
    writer.block_end();

    const cldnn::custom_gpu_primitive op_slice(output_name,
                                               {input_name},
                                               {writer.get_code()},
                                               entry_point_name,
                                               get_kernel_args(1, 1),
                                               "",
                                               layout,
                                               {1});
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

    writer << "__kernel void " << entry_point_name << "(const __global char input0"
           << array_dims(input0_shape) << ", const __global float input1"
           << array_dims(input1_shape) << ", const __global float input2"
           << array_dims(input2_shape) << ", __global float output" << array_dims(output_shape)
           << ")\n";

    writer.block_begin();
    {
        size_t var_idx = 0;
        // Main loops
        for (auto const& i : output_shape)
        {
            writer << "for (uint i" << var_idx << " = 0; i" << var_idx << " < " << i << "; ++i"
                   << var_idx << ")\n";
            writer.block_begin();
            ++var_idx;
        }

        writer << "if (input0" << access_dims(input0_shape) << " != 0)\n";
        writer.block_begin();
        {
            writer << "output" << access_dims(output_shape) << " = input1"
                   << access_dims(input1_shape) << ";\n";
        }
        writer.block_end();
        writer << "else\n";
        writer.block_begin();
        {
            writer << "output" << access_dims(output_shape) << " = input2"
                   << access_dims(input2_shape) << ";\n";
        }
        writer.block_end();

        // Closing brackets for main loops
        for (auto const& i : output_shape)
        {
            writer.block_end();
        }
    }
    writer.block_end();

    const cldnn::custom_gpu_primitive op_select(output_name,
                                                {input0_name, input1_name, input2_name},
                                                {writer.get_code()},
                                                entry_point_name,
                                                get_kernel_args(3, 1),
                                                "",
                                                layout,
                                                {1});
    topology.add(op_select);
}

void runtime::intelgpu::do_logic_kernel(cldnn::topology& topology,
                                        const string& inputA_name,
                                        const Shape& inputA_shape,
                                        const string& inputA_type,
                                        const string& inputB_name,
                                        const Shape& inputB_shape,
                                        const string& inputB_type,
                                        const string& output_name,
                                        const Shape& output_shape,
                                        const element::Type& output_type,
                                        const string& operation)
{
    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const string entry_point_name = "logic_" + output_name;
    codegen::CodeWriter writer;

    writer << "__kernel void " << entry_point_name << "(const __global " << inputA_type << " inputA"
           << array_dims(inputA_shape) << ", const __global " << inputB_type << " inputB"
           << array_dims(inputB_shape) << ", __global char output" << array_dims(output_shape)
           << ")\n";

    writer.block_begin();
    {
        size_t var_idx = 0;
        // Main loops
        for (auto const& i : output_shape)
        {
            writer << "for (uint i" << var_idx << " = 0; i" << var_idx << " < " << i << "; ++i"
                   << var_idx << ")\n";
            writer.block_begin();
            ++var_idx;
        }

        writer << "if (inputA" << access_dims(inputA_shape) << operation << "inputB"
               << access_dims(inputB_shape) << ")\n";

        writer.block_begin();
        {
            writer << "output" << access_dims(output_shape) << " = 1;\n";
        }
        writer.block_end();
        writer << "else\n";
        writer.block_begin();
        {
            writer << "output" << access_dims(output_shape) << " = 0;\n";
        }
        writer.block_end();

        // Closing brackets for main loops
        for (auto const& i : output_shape)
        {
            writer.block_end();
        }
    }
    writer.block_end();

    const cldnn::custom_gpu_primitive op_logical(output_name,
                                                 {inputA_name, inputB_name},
                                                 {writer.get_code()},
                                                 entry_point_name,
                                                 get_kernel_args(2, 1),
                                                 "",
                                                 layout,
                                                 {1});
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

    writer << "__kernel void " << entry_point_name << "(const __global float input"
           << array_dims(input_shape) << ", __global float output" << array_dims(output_shape)
           << ")\n";

    writer.block_begin();
    {
        gws = generate_loops(writer, output_shape, true);

        writer << "output" << access_dims(output_shape) << " = input"
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

    writer << "__kernel void " << entry_point_name << "(const __global " << input_type_name
           << " input" << array_dims(input_shape) << ", __global " << output_type_name << " output"
           << array_dims(output_shape) << ")\n";

    writer.block_begin();
    {
        gws = generate_loops(writer, output_shape, true);

        writer << "output" << access_dims(output_shape) << " = convert_" << output_type_name
               << "(input" << access_dims(output_shape) << ");\n";

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
