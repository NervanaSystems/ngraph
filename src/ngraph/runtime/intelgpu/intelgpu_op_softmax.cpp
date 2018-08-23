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

#include "ngraph/runtime/intelgpu/code_writer.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_layout.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_custom_kernels.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_softmax.hpp"

using namespace std;
using namespace ngraph;

static Shape shape_dims(const Shape& dimentions, const AxisSet& axis = {})
{
    size_t var_idx = 0;
    Shape output_shape;
    for (auto const& dim : dimentions)
    {
        if (axis.find(var_idx) == axis.end())
        {
            output_shape.push_back(dim);
        }
        ++var_idx;
    }

    if (output_shape.size() == 0)
    { // it means scalar
        output_shape.push_back(1);
    }

    return output_shape;
}

static vector<size_t> generate_loops_w_axes(codegen::CodeWriter& writer,
                                            const Shape& shape,
                                            bool is_begin,
                                            const AxisSet& axis,
                                            const string& expression)
{
    const size_t cldnn_gws_lim = 3;
    vector<size_t> gws;
    size_t var_idx = 0;
    size_t dim_idx = 0;

    for (auto const& i : shape)
    {
        if (axis.find(var_idx) == axis.end())
        {
            if (dim_idx < cldnn_gws_lim)
            {
                if (is_begin)
                {
                    writer << "const unsigned i" << var_idx << " = get_global_id(" << dim_idx
                           << ");\n";
                    gws.push_back(i);
                }
                ++dim_idx;
            }
            else
            {
                if (is_begin)
                {
                    writer << "for (uint i" << var_idx << " = 0; i" << var_idx << " < " << i
                           << "; ++i" << var_idx << ")\n";
                    writer.block_begin();
                }
                else
                {
                    writer.block_end();
                }
            }
        }
        ++var_idx;
    }
    if (is_begin)
    {
        writer << expression;
    }

    var_idx = 0;

    for (auto const& i : shape)
    {
        if (axis.find(var_idx) != axis.end())
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

void runtime::intelgpu::do_softmax_operation(cldnn::topology& topology,
                                             const string& input_name,
                                             const Shape& input_shape,
                                             const element::Type& input_type,
                                             const string& output_name,
                                             const Shape& output_shape,
                                             const element::Type& output_type,
                                             const AxisSet& axes)
{
    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const string entry_point_name = "softmax_" + output_name;
    const string middle_name = entry_point_name + "_middle";
    const string entry_point_middle_name = "softmax_middle_" + output_name;
    const string expression = "output" + access_dims(input_shape, axes) + " = 0.0f;\n";
    const Shape new_shape = shape_dims(output_shape, axes);
    const cldnn::layout layout_middle = IntelGPULayout::create_cldnn_layout(output_type, new_shape);
    codegen::CodeWriter writer0;
    codegen::CodeWriter writer1;
    vector<size_t> gws;

    writer0 << "__kernel void " << entry_point_middle_name << "(const __global float input"
            << array_dims(input_shape) << ", __global float output" << array_dims(input_shape, axes)
            << ")\n";

    writer0.block_begin();
    {
        gws = generate_loops_w_axes(writer0, output_shape, true, axes, expression);

        writer0 << "output" << access_dims(input_shape, axes) << " += exp(input"
                << access_dims(input_shape) << ");\n";

        generate_loops_w_axes(writer0, output_shape, false, axes, "");
    }
    writer0.block_end();

    const cldnn::custom_gpu_primitive op_softmax_middle(middle_name,
                                                        {input_name},
                                                        {writer0.get_code()},
                                                        entry_point_middle_name,
                                                        get_kernel_args(1, 1),
                                                        "",
                                                        layout_middle,
                                                        gws);
    topology.add(op_softmax_middle);

    writer1 << "__kernel void " << entry_point_name << "(const __global float input0"
            << array_dims(input_shape) << ", const __global float input1"
            << array_dims(input_shape, axes) << ", __global float output"
            << array_dims(output_shape) << ")\n";

    writer1.block_begin();
    {
        gws = generate_loops(writer1, output_shape, true);
        writer1 << "output" << access_dims(input_shape) << " = exp(input0"
                << access_dims(input_shape) << ")/input1" << access_dims(input_shape, axes)
                << ";\n";
        generate_loops(writer1, output_shape, false);
    }
    writer1.block_end();

    const cldnn::custom_gpu_primitive op_softmax(output_name,
                                                 {input_name, middle_name},
                                                 {writer1.get_code()},
                                                 entry_point_name,
                                                 get_kernel_args(2, 1),
                                                 "",
                                                 layout,
                                                 gws);
    topology.add(op_softmax);
}
