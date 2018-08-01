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
#include "ngraph/runtime/intelgpu/intelgpu_op_custom_kernels.hpp"

#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

static vector<cldnn_arg> parameters_2inp_1out = {{arg_input, 0}, {arg_input, 1}, {arg_output, 0}};

static string array_dims(const Shape& dimentions)
{
    string buffer;

    for (auto const& dim : dimentions)
    {
        buffer += "[" + to_string(dim) + "]";
    }

    return buffer;
}

static string access_dims(const Shape& dimentions, const AxisSet& axis = {})
{
    size_t var_idx = 0;
    string buffer;

    for (auto const& i : dimentions)
    {
        if (axis.find(var_idx) == axis.end())
        {
            buffer += "[i" + to_string(var_idx) + "]";
        }
        ++var_idx;
    }

    return buffer;
}

static string
    access_dims_strided(const Shape& dimentions, const Shape& pad_below, const Shape& pad_interior)
{
    string buffer;
    size_t var_idx = 0;

    for (auto const& i : dimentions)
    {
        buffer += "[i" + to_string(var_idx) + " * (" + to_string(pad_interior.at(var_idx)) +
                  " + 1) + " + to_string(pad_below.at(var_idx)) + "]";
        ++var_idx;
    }

    return buffer;
}

static void do_dot_operation_error(const Shape& shapeA, const Shape& shapeB, const Shape& shapeZ)
{
    throw invalid_argument("IntelGPU Dot operation. Conbination ShapeA" + array_dims(shapeA) +
                           ", ShapeB" + array_dims(shapeB) + ", ShapeOutput" + array_dims(shapeZ) +
                           " is not supported.");
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
    const string entry_point_name = "op_pad_kernel";
    codegen::CodeWriter writer;

    // The kernel name and parameters
    writer << "__kernel void " << entry_point_name << "(const __global float input"
           << array_dims(input_shape) << ", const __global float scalar[1], __global float output"
           << array_dims(output_shape) << ")\n";

    writer.block_begin();
    {
        // Loop for Broadcast scalar over full output tensor
        size_t var_idx = 0;
        for (auto const& i : output_shape)
        {
            writer << "for (uint i" << var_idx << " = 0; i" << var_idx << " < " << i << "; ++i"
                   << var_idx << ")\n";
            writer.block_begin();
            ++var_idx;
        }

        writer << "output" << access_dims(output_shape) << " = scalar[0];\n";

        // Closing brackets for Broadcast loop
        for (auto const& i : output_shape)
        {
            writer.block_end();
        }

        // Loop for Copy input matrix into output matrix with padding.
        // Padding include "pad_below" and "pad_interior" according nGraph documentation
        var_idx = 0;
        for (auto const& i : input_shape)
        {
            writer << "for (uint i" << var_idx << " = 0; i" << var_idx << " < " << i << "; ++i"
                   << var_idx << ")\n";
            writer.block_begin();
            ++var_idx;
        }

        writer << "output" << access_dims_strided(input_shape, pad_below, pad_interior)
               << " = input" << access_dims(input_shape) << ";\n";

        // Closing brackets for main Copy loop
        for (auto const& i : input_shape)
        {
            writer.block_end();
        }

    } // End of function bracket
    writer.block_end();

    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const cldnn::custom_gpu_primitive op_scalar(output_name,
                                                {input_name, scalar_name},
                                                {writer.get_code()},
                                                entry_point_name,
                                                parameters_2inp_1out,
                                                "",
                                                layout);
    topology.add(op_scalar);
}

static void do_1d_scalar_mul(codegen::CodeWriter& writer,
                             string& kernel_name,
                             const Shape& shapeA,
                             const Shape& shapeB)
{
    const size_t countA = shapeA.empty() ? 0 : shape_size<Shape>(shapeA);
    const size_t countB = shapeB.empty() ? 0 : shape_size<Shape>(shapeB);
    const size_t countZ = max(countA, countB);
    kernel_name = "do_1d_scalar_mul";

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

static void do_2d_2d_mul(codegen::CodeWriter& writer,
                         string& kernel_name,
                         const Shape& shapeA,
                         const Shape& shapeB)
{
    const size_t rows = shapeA.at(0);
    const size_t colrow = shapeA.at(1);
    const size_t cols = shapeB.back();
    kernel_name = "do_2d_2d_mul";

    writer << "__kernel void " << kernel_name << "(const __global float inputA"
           << array_dims(shapeA) << ", const __global float inputB" << array_dims(shapeB)
           << ", __global float output" << array_dims({rows, cols}) << ")\n";
    writer.block_begin();
    {
        size_t var_idx = 0;
        // Main loops
        for (auto const& i : shapeA)
        {
            writer << "for (uint i" << var_idx << " = 0; i" << var_idx << " < " << i << "; ++i"
                   << var_idx << ")\n";
            writer.block_begin();
            ++var_idx;
        }

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
        for (auto const& i : shapeA)
        {
            writer.block_end();
        }
    }
    writer.block_end();
}

static void do_3d_3d_mul(codegen::CodeWriter& writer,
                         string& kernel_name,
                         const Shape& shapeA,
                         const Shape& shapeB,
                         const Shape& shapeZ)
{
    const size_t colrow = shapeA.back();
    kernel_name = "do_3d_3d_mul";

    writer << "__kernel void " << kernel_name << "(const __global float inputA"
           << array_dims(shapeA) << ", const __global float inputB" << array_dims(shapeB)
           << ", __global float output" << array_dims(shapeZ) << ")\n";
    writer.block_begin();
    {
        size_t var_idx = 0;
        // Main loops
        for (auto const& i : shapeZ)
        {
            writer << "for (uint i" << var_idx << " = 0; i" << var_idx << " < " << i << "; ++i"
                   << var_idx << ")\n";
            writer.block_begin();
            ++var_idx;
        }

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
        for (auto const& i : shapeZ)
        {
            writer.block_end();
        }
    }
    writer.block_end();
}

static void do_3d_2d_mul(codegen::CodeWriter& writer,
                         string& kernel_name,
                         const Shape& shapeA,
                         const Shape& shapeB,
                         const Shape& shapeZ)
{
    const size_t colrow = shapeA.back();
    kernel_name = "do_3d_2d_mul";

    writer << "__kernel void " << kernel_name << "(const __global float inputA"
           << array_dims(shapeA) << ", const __global float inputB" << array_dims(shapeB)
           << ", __global float output" << array_dims(shapeZ) << ")\n";
    writer.block_begin();
    {
        size_t var_idx = 0;
        // Main loops
        for (auto const& i : shapeZ)
        {
            writer << "for (uint i" << var_idx << " = 0; i" << var_idx << " < " << i << "; ++i"
                   << var_idx << ")\n";
            writer.block_begin();
            ++var_idx;
        }

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
        for (auto const& i : shapeZ)
        {
            writer.block_end();
        }
    }
    writer.block_end();
}

static void do_2d_1d_mul(codegen::CodeWriter& writer,
                         string& kernel_name,
                         const Shape& shapeA,
                         const Shape& shapeB)
{
    const size_t rows = shapeA.at(0);
    const size_t colrow = shapeA.at(1);
    kernel_name = "do_2d_1d_mul";

    writer << "__kernel void " << kernel_name << "(const __global float inputA"
           << array_dims(shapeA) << ", const __global float inputB" << array_dims(shapeB)
           << ", __global float output" << array_dims({rows}) << ")\n";
    writer.block_begin();
    {
        writer << "for (uint i0 = 0; i0 < " << rows << "; ++i0)\n";
        writer.block_begin();
        {
            writer << "float sum = 0.0f;\n";
            writer << "for (uint i1 = 0; i1 < " << colrow << "; ++i1)\n";
            writer.block_begin();
            {
                writer << "sum += inputA[i0][i1] * inputB[i1];\n";
            }
            writer.block_end();
            writer << "output[i0] = sum;\n";
        }
        writer.block_end();
    }
    writer.block_end();
}

static void do_scalar_scalar_mul(codegen::CodeWriter& writer, string& kernel_name)
{
    kernel_name = "scalar_scalar_mul";

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
        throw invalid_argument("do_1d_1d_mul: Shape" + array_dims(shape) + " must be 1D");
    }

    const size_t& size = shape.front();
    kernel_name = "do_1d_1d_mul";

    writer << "__kernel void " << kernel_name << "(const __global float inputA" << array_dims(shape)
           << ", const __global float inputB" << array_dims(shape)
           << ", __global float output[1])\n";
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
    string entry_point_name = "dot_unknown";
    codegen::CodeWriter writer;

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
            do_2d_1d_mul(writer, entry_point_name, inputA_shape, inputB_shape);
        }
        else if (inputA_shape.size() == 2 && inputB_shape.size() == 2)
        {
            do_2d_2d_mul(writer, entry_point_name, inputA_shape, inputB_shape);
        }
        else if (inputA_shape.size() == 3 && inputB_shape.size() == 3)
        {
            do_3d_3d_mul(writer, entry_point_name, inputA_shape, inputB_shape, output_shape);
        }
        else if (inputA_shape.size() == 3 && inputB_shape.size() == 2)
        {
            do_3d_2d_mul(writer, entry_point_name, inputA_shape, inputB_shape, output_shape);
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

    //cout << writer.get_code() << endl;
    const cldnn::custom_gpu_primitive op_dot(output_name,
                                             {inputA_name, inputB_name},
                                             {writer.get_code()},
                                             entry_point_name,
                                             parameters_2inp_1out,
                                             "",
                                             layout);
    topology.add(op_dot);
}
