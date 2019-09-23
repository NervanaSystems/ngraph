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

using namespace std;
using namespace ngraph;
using namespace ngraph::runtime::intelgpu;

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

CustomKernels::krnl_info CustomKernels::build_krnl(const shared_ptr<op::Softmax>& op) const
{
    const string& input_name = op->get_input_tensor_name(0);
    const Shape& input_shape = op->get_input_shape(0);
    const element::Type& input_type = op->get_input_element_type(0);
    const string& output_name = op->get_output_tensor_name(0);
    const Shape& output_shape = op->get_output_shape(0);
    const element::Type& output_type = op->get_output_element_type(0);
    const AxisSet& axes = op->get_axes();
    const string entry_point_name = "softmax_" + output_name;
    const string middle_name = entry_point_name + "_middle";
    const string entry_point_middle_name = "softmax_middle_" + output_name;
    const string expression = "output" + access_dims(input_shape, "i", axes) + " = 0.0f;\n";
    const Shape new_shape = shape_dims(output_shape, axes);
    CodeWriter writer0;
    CodeWriter writer1;
    vector<size_t> gws;

    writer0 << "__kernel void " << entry_point_middle_name << "(const __global "
            << get_opencl_type_name(input_type) << " input" << array_dims(input_shape)
            << ", __global " << get_opencl_type_name(output_type) << " output"
            << array_dims(input_shape, axes) << ")\n";

    writer0.block_begin();
    {
        gws = generate_loops_w_axes(writer0, output_shape, true, axes, expression);

        writer0 << "output" << access_dims(input_shape, "i", axes) << " += exp(input"
                << access_dims(input_shape) << ");\n";

        generate_loops_w_axes(writer0, output_shape, false, axes, "");
    }
    writer0.block_end();

    const CustomKernelInfo op_softmax_middle(middle_name,
                                             new_shape,
                                             output_type,
                                             {input_name},
                                             {writer0.get_code()},
                                             entry_point_middle_name,
                                             gws);

    writer1 << "__kernel void " << entry_point_name << "(const __global "
            << get_opencl_type_name(input_type) << " input0" << array_dims(input_shape)
            << ", const __global " << get_opencl_type_name(input_type) << " input1"
            << array_dims(input_shape, axes) << ", __global " << get_opencl_type_name(output_type)
            << " output" << array_dims(output_shape) << ")\n";

    writer1.block_begin();
    {
        gws = generate_loops(writer1, output_shape, true);
        writer1 << "output" << access_dims(input_shape) << " = exp(input0"
                << access_dims(input_shape) << ")/input1" << access_dims(input_shape, "i", axes)
                << ";\n";
        generate_loops(writer1, output_shape, false);
    }
    writer1.block_end();

    const CustomKernelInfo op_softmax(output_name,
                                      output_shape,
                                      output_type,
                                      {input_name, middle_name},
                                      {writer1.get_code()},
                                      entry_point_name,
                                      gws);
    return {op_softmax_middle, op_softmax};
}
