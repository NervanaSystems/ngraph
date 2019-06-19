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

#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;
using namespace ngraph::runtime::intelgpu;

static CustomKernels::krnl_info do_all_any_op(const shared_ptr<op::util::LogicalReduction>& op,
                                              const string& operation)
{
    const string& input0_name = op->get_input_tensor_name(0);
    const Shape& input0_shape = op->get_input_shape(0);
    const string& output_name = op->get_output_tensor_name(0);
    const Shape& output_shape = op->get_output_shape(0);
    const element::Type& output_type = op->get_output_element_type(0);
    const AxisSet& axis = op->get_reduction_axes();
    const shared_ptr<Node> def_val = op->get_default_value();
    const shared_ptr<op::Constant> def_const = static_pointer_cast<op::Constant>(def_val);
    const vector<string>& values = def_const->get_value_strings();
    const string& init_val = values.at(0);
    const string entry_point_name = "custom_op_all_any_" + output_name;
    const string kernel_type_name = get_opencl_type_name(output_type);
    const size_t input_size = shape_size<Shape>(input0_shape);
    CodeWriter writer;

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

    const CustomKernelInfo krn_ret(output_name,
                                   output_shape,
                                   output_type,
                                   {input0_name},
                                   {writer.get_code()},
                                   entry_point_name);
    return {krn_ret};
}

CustomKernels::krnl_info CustomKernels::build_krnl(const shared_ptr<op::All>& op) const
{
    return do_all_any_op(op, "lhs && rhs");
}

CustomKernels::krnl_info CustomKernels::build_krnl(const shared_ptr<op::Any>& op) const
{
    return do_all_any_op(op, "lhs || rhs");
}
