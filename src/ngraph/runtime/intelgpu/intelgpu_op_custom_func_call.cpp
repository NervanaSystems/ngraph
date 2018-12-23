//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include <CPP/custom_gpu_primitive.hpp>

#include "ngraph/runtime/intelgpu/code_writer.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_layout.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_custom_func_call.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_custom_kernels.hpp"

using namespace std;
using namespace ngraph;

void runtime::intelgpu::do_all_any_op(cldnn::topology& topology,
                                      const string& input0_name,
                                      const Shape& input0_shape,
                                      const string& output_name,
                                      const Shape& output_shape,
                                      const element::Type& output_type,
                                      const AxisSet& axis,
                                      const std::string& operation,
                                      const std::string& init_val)
{
    const string entry_point_name = "custom_op_all_any_" + output_name;
    const string kernel_type_name = get_opencl_type_name(output_type);
    const size_t input_size = shape_size<Shape>(input0_shape);
    codegen::CodeWriter writer;

    // The kernel name and parameters
    gen_func_def(writer,
                 entry_point_name,
                 {1, kernel_type_name},
                 {input0_shape, {1}},
                 kernel_type_name,
                 output_shape);

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

        writer << "output" << access_dims(output_shape) << " = " << init_val << ";\n";

        // Closing brackets for initialization loop
        for (auto const& i : output_shape)
        {
            writer.block_end();
        }

        if (input_size && !input0_shape.empty())
        {
            // Main operation loop
            var_idx = 0;
            for (auto const& i : input0_shape)
            {
                writer << "for (uint i" << var_idx << " = 0; i" << var_idx << " < " << i << "; ++i"
                       << var_idx << ")\n";
                writer.block_begin();
                ++var_idx;
            }

            writer << kernel_type_name << " lhs = output" << access_dims(input0_shape, "i", axis)
                   << ";\n"
                   << kernel_type_name << " rhs = input0" << access_dims(input0_shape) << ";\n"
                   << "output" << access_dims(input0_shape, "i", axis) << " = (" << operation
                   << ");\n";

            // Closing brackets for loop
            for (auto const& i : input0_shape)
            {
                writer.block_end();
            }
        }
    } // End of function bracket
    writer.block_end();

    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const cldnn::custom_gpu_primitive op_all_any(output_name,
                                                 {input0_name},
                                                 {writer.get_code()},
                                                 entry_point_name,
                                                 get_kernel_args(1, 1),
                                                 "",
                                                 layout,
                                                 {1});
    topology.add(op_all_any);
}

static void get_custom_func_name(codegen::CodeWriter& writer,
                                 vector<shared_ptr<Function>>& func,
                                 const string& func_name,
                                 const string& type_name)
{
    if (func.size() != 1)
    {
        throw invalid_argument("IntelGPU Custom_Call operation. Custom function number: " +
                               to_string(func.size()) + " expected: 1");
    }

    writer << type_name << " " << func_name << "(const " << type_name << " input0, const "
           << type_name << " input1)\n";
    writer.block_begin();
    {
        for (shared_ptr<Node> op : func.at(0)->get_ordered_ops())
        {
            if ((op->description() != "Parameter") && (op->description() != "Result"))
            {
                if (op->description() == "Multiply")
                {
                    writer << "return input0 * input1;\n";
                }
                else if (op->description() == "Add")
                {
                    writer << "return input0 + input1;\n";
                }
                else if (op->description() == "Maximum")
                {
                    writer << "return max(input0, input1);\n";
                }
                else if (op->description() == "Minimum")
                {
                    writer << "return min(input0, input1);\n";
                }
                else if (op->description() == "And")
                {
                    writer << "return input0 && input1;\n";
                }
                else if (op->description() == "Or")
                {
                    writer << "return input0 || input1;\n";
                }
                else if (op->description() == "Equal")
                {
                    writer << "return input0 == input1;\n";
                }
                else if (op->description() == "NotEqual")
                {
                    writer << "return input0 != input1;\n";
                }
                else
                {
                    writer << "UNIMPLEMENTED_FUNCTION_INTELGPU: " << op->description() << "\n";
                }
            }
        }

    } // End of function bracket
    writer.block_end();
}

void runtime::intelgpu::do_reduce_func_call(cldnn::topology& topology,
                                            const string& input0_name,
                                            const Shape& input0_shape,
                                            const string& input1_name,
                                            const Shape& input1_shape,
                                            const string& output_name,
                                            const Shape& output_shape,
                                            const element::Type& output_type,
                                            const AxisSet& axis,
                                            vector<shared_ptr<Function>>& func)
{
    const string entry_point_name = "reduce_func_call_" + output_name;
    const string aux_point_name = "aux_call_" + output_name;
    const string kernel_type_name = get_opencl_type_name(output_type);
    const size_t input_size = shape_size<Shape>(input0_shape);
    codegen::CodeWriter writer;

    get_custom_func_name(writer, func, aux_point_name, kernel_type_name);
    // The kernel name and parameters
    gen_func_def(writer,
                 entry_point_name,
                 {2, kernel_type_name},
                 {input0_shape, {1}},
                 kernel_type_name,
                 output_shape);

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

        writer << "output" << access_dims(output_shape) << " = input1" << access_dims(input1_shape)
               << ";\n";

        // Closing brackets for initialization loop
        for (auto const& i : output_shape)
        {
            writer.block_end();
        }

        if (input_size && !input0_shape.empty())
        {
            // Main operation loop
            var_idx = 0;
            for (auto const& i : input0_shape)
            {
                writer << "for (uint i" << var_idx << " = 0; i" << var_idx << " < " << i << "; ++i"
                       << var_idx << ")\n";
                writer.block_begin();
                ++var_idx;
            }

            writer << "output" << access_dims(input0_shape, "i", axis) << " = " << aux_point_name
                   << "(output" << access_dims(input0_shape, "i", axis) << ", input0"
                   << access_dims(input0_shape) << ");\n";

            // Closing brackets for loop
            for (auto const& i : input0_shape)
            {
                writer.block_end();
            }
        }
    } // End of function bracket
    writer.block_end();

    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const cldnn::custom_gpu_primitive op_product(output_name,
                                                 {input0_name, input1_name},
                                                 {writer.get_code()},
                                                 entry_point_name,
                                                 get_kernel_args(2, 1),
                                                 "",
                                                 layout,
                                                 {1});
    topology.add(op_product);
}
