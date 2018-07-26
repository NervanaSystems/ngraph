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

#include <CPP/custom_gpu_primitive.hpp>
#include <CPP/reshape.hpp>

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

    for (auto i = dimentions.cbegin(); i != dimentions.cend(); ++i, ++var_idx)
    {
        if (axis.find(var_idx) == axis.end())
        {
            buffer += "[i" + to_string(var_idx) + "]";
        }
    }

    return buffer;
}

static string
    access_dims_strided(const Shape& dimentions, const Shape& pad_below, const Shape& pad_interior)
{
    string buffer;
    size_t var_idx = 0;

    for (auto i = dimentions.cbegin(); i != dimentions.cend(); ++i, ++var_idx)
    {
        buffer += "[i" + to_string(var_idx) + " * (" + to_string(pad_interior.at(var_idx)) +
                  " + 1) + " + to_string(pad_below.at(var_idx)) + "]";
    }

    return buffer;
}

void runtime::intelgpu::do_pad_kernel(cldnn::topology& topology,
                                      const string& input_name,
                                      const Shape& input_shape,
                                      const string& scalar_name,
                                      const string& output_name,
                                      const Shape& output_shape,
                                      const element::Type& output_type,
                                      const Shape& pad_below,
                                      const Shape& pad_interior)
{
    const size_t input_count = shape_size<Shape>(output_shape);
    const string entry_point_name = "op_pad_kernel";
    ostringstream kernel_code;

    // The kernel name and parameters
    kernel_code << "__kernel void " << entry_point_name << "(const __global float input"
                << array_dims(input_shape)
                << ", const __global float scalar[1], __global float output"
                << array_dims(output_shape) << ")\n{\n";

    // Loop for Broadcast scalar over full output tensor
    size_t var_idx = 0;
    for (auto i = output_shape.cbegin(); i != output_shape.cend(); ++i, ++var_idx)
    {
        kernel_code << "for (uint i" << var_idx << " = 0; i" << var_idx << " < " << *i << "; ++i"
                    << var_idx << ") {\n";
    }
    kernel_code << "output" << access_dims(output_shape) << " = scalar[0];\n";
    // Closing brackets for Broadcast loop
    kernel_code << string(output_shape.size(), '}') << "\n\n";

    // Loop for Copy input matrix into output matrix with padding.
    // Padding include "pad_below" and "pad_interior" according nGraph documentation
    var_idx = 0;
    for (auto i = input_shape.cbegin(); i != input_shape.cend(); ++i, ++var_idx)
    {
        kernel_code << "for (uint i" << var_idx << " = 0; i" << var_idx << " < " << *i << "; ++i"
                    << var_idx << ") {\n";
    }
    kernel_code << "output" << access_dims_strided(input_shape, pad_below, pad_interior)
                << " = input" << access_dims(input_shape) << ";\n";

    // Closing brackets for main Copy loop
    kernel_code << string(input_shape.size(), '}');
    // End of function bracket
    kernel_code << "\n}\n";

    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const cldnn::custom_gpu_primitive op_scalar(output_name,
                                                {input_name, scalar_name},
                                                {kernel_code.str()},
                                                entry_point_name,
                                                parameters_2inp_1out,
                                                "",
                                                layout,
                                                {1});
    topology.add(op_scalar);
}
