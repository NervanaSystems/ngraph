// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------
#include <algorithm>

#include "ngraph/codegen/code_writer.hpp"
#include "ngraph/runtime/cpu/cpu_kernel_emitters.hpp"
#include "ngraph/runtime/cpu/cpu_kernel_utils.hpp"

using namespace ngraph;
using namespace ngraph::runtime::cpu::kernels;

template <typename T>
std::string emit_bracketed_string(std::vector<T> data)
{
    std::stringstream ss;

    if (data.size() == 0)
        return "";

    for (auto s : data)
    {
        ss << "[" << s << "]";
    }

    return ss.str();
}

void ngraph::runtime::cpu::kernels::emit_broadcast(codegen::CodeWriter& writer,
                                                   const std::string& element_type,
                                                   const std::string& arg0, // replacement context
                                                   const std::string& out,
                                                   const Shape& arg0_shape,
                                                   const Shape& out_shape,
                                                   const AxisSet& broadcast_axes)
{
    std::string source_nd_name = writer.generate_temporary_name("source_nd");
    std::string dest_nd_name = writer.generate_temporary_name("dest_nd");

    writer << element_type << "(&" << source_nd_name << ")" << emit_bracketed_string(arg0_shape)
           << " = *reinterpret_cast<" << element_type << "(*)" << emit_bracketed_string(arg0_shape)
           << ">(" << arg0 << ");\n";
    writer << element_type << "(&" << dest_nd_name << ")" << emit_bracketed_string(out_shape)
           << " = *reinterpret_cast<" << element_type << "(*)" << emit_bracketed_string(out_shape)
           << ">(" << out << ");\n";

    std::vector<std::string> index_vars;
    for (size_t i = 0; i < out_shape.size(); i++)
    {
        std::string index_var = writer.generate_temporary_name("i");

        writer << start_index_loop(index_var, 0, out_shape[i], i == 0);
        writer.indent++;

        index_vars.push_back(index_var);
    }

    std::vector<std::string> source_indexes;
    for (size_t i = 0; i < out_shape.size(); ++i)
    {
        if (broadcast_axes.count(i) == 0)
        {
            source_indexes.push_back(index_vars[i]);
        }
    }

    writer << dest_nd_name << emit_bracketed_string(index_vars) << " = " << source_nd_name
           << emit_bracketed_string(source_indexes) << ";\n";

    for (size_t i = out_shape.size(); i-- > 0;)
    {
        writer.indent--;
        writer << end_index_loop(index_vars[i]);
    }
}

//
// For the reference kernel this is based on, see ngraph/runtime/kernel/concat.hpp.
//
void ngraph::runtime::cpu::kernels::emit_concat(codegen::CodeWriter& writer,
                                                const std::string& element_type,
                                                const std::vector<std::string>& args,
                                                const std::string& out,
                                                const std::vector<Shape>& in_shapes,
                                                const Shape& out_shape,
                                                size_t concatenation_axis)
{
    size_t concatenation_pos = 0;

    for (size_t i = 0; i < args.size(); i++)
    {
        Coordinate out_start_coord = Coordinate(out_shape.size(), 0);
        out_start_coord[concatenation_axis] = concatenation_pos;

        Coordinate out_end_coord = out_shape;
        out_end_coord[concatenation_axis] = concatenation_pos + in_shapes[i][concatenation_axis];

        CoordinateTransform input_transform(in_shapes[i]);
        CoordinateTransform output_chunk_transform(out_shape, out_start_coord, out_end_coord);

        emit_pointwise_copy(
            writer, element_type, args[i], out, input_transform, output_chunk_transform);

        concatenation_pos += in_shapes[i][concatenation_axis];
    }
}

void ngraph::runtime::cpu::kernels::emit_replace_slice(
    codegen::CodeWriter& writer,
    const std::string& element_type,
    const std::string& arg0, // replacement context
    const std::string& arg1, // replacement value
    const std::string& out,
    const Shape& arg1_shape,
    const Shape& out_shape,
    const Coordinate& lower_bounds,
    const Coordinate& upper_bounds,
    const Strides& strides)
{
    // Step 1: Copy the entire replacement context to the output.
    CoordinateTransform copy_transform(out_shape);
    emit_pointwise_copy(writer, element_type, arg0, out, copy_transform, copy_transform);

    // Step 2: Overwrite the slice for replacement.
    CoordinateTransform input_transform(arg1_shape);
    CoordinateTransform output_transform(out_shape, lower_bounds, upper_bounds, strides);

    emit_pointwise_copy(writer, element_type, arg1, out, input_transform, output_transform);
}

void ngraph::runtime::cpu::kernels::emit_slice(codegen::CodeWriter& writer,
                                               const std::string& element_type,
                                               const std::string& arg0, // replacement context
                                               const std::string& out,
                                               const Shape& arg0_shape,
                                               const Shape& out_shape,
                                               const Coordinate& lower_bounds,
                                               const Coordinate& upper_bounds,
                                               const Strides& strides)
{
    std::vector<std::string> index_vars;

    std::string source_nd_name = writer.generate_temporary_name("source_nd");
    std::string dest_nd_name = writer.generate_temporary_name("dest_nd");

    writer << element_type << "(&" << source_nd_name << ")" << emit_bracketed_string(arg0_shape)
           << " = *reinterpret_cast<" << element_type << "(*)" << emit_bracketed_string(arg0_shape)
           << ">(" << arg0 << ");\n";
    writer << element_type << "(&" << dest_nd_name << ")" << emit_bracketed_string(out_shape)
           << " = *reinterpret_cast<" << element_type << "(*)" << emit_bracketed_string(out_shape)
           << ">(" << out << ");\n";

    for (size_t i = 0; i < out_shape.size(); i++)
    {
        std::string index_var = writer.generate_temporary_name("i");

        writer << start_index_loop(index_var, 0, out_shape[i], i == 0);
        writer.indent++;

        index_vars.push_back(index_var);
    }

    std::vector<std::string> source_indexes;
    size_t j = 0;
    for (size_t i = 0; i < lower_bounds.size(); ++i)
    {
        if (lower_bounds[i] == upper_bounds[i])
        {
            source_indexes.push_back(std::to_string(lower_bounds[i]));
        }
        else
        {
            std::stringstream ss;
            ss << lower_bounds[i];
            ss << " + " << index_vars[j];
            ss << " * " << strides[i];
            source_indexes.push_back(ss.str());
            j += 1;
        }
    }

    writer << dest_nd_name << emit_bracketed_string(index_vars) << " = " << source_nd_name
           << emit_bracketed_string(source_indexes) << ";\n";

    for (size_t i = out_shape.size(); i-- > 0;)
    {
        writer.indent--;
        writer << end_index_loop(index_vars[i]);
    }
}

void ngraph::runtime::cpu::kernels::emit_reshape(codegen::CodeWriter& writer,
                                                 const std::string& element_type,
                                                 const std::string& arg0, // replacement context
                                                 const std::string& out,
                                                 const Shape& arg0_shape,
                                                 const Shape& out_shape,
                                                 const AxisVector& arg0_axis_order)
{
    for (auto x : arg0_axis_order)
        std::cout << x << ",";
    std::cout << std::endl;
    Shape in_start_corner(arg0_shape.size(), 0); // (0,...0)
    Shape in_strides(arg0_shape.size(), 1);      // (1,...,1)

    CoordinateTransform input_transform(
        arg0_shape, in_start_corner, arg0_shape, in_strides, arg0_axis_order);

    CoordinateTransform output_transform(out_shape);
    std::cout << "emit_pointwise_copy" << std::endl;
    emit_pointwise_copy(writer, element_type, arg0, out, input_transform, output_transform);
}

void ngraph::runtime::cpu::kernels::emit_sum(codegen::CodeWriter& writer,
                                             const std::string& element_type,
                                             const std::string& arg0, // replacement context
                                             const std::string& out,
                                             const Shape& arg0_shape,
                                             const Shape& out_shape,
                                             const AxisSet& reduction_axes)
{
    std::string source_nd_name = writer.generate_temporary_name("source_nd");
    std::string dest_nd_name = writer.generate_temporary_name("dest_nd");

    writer << element_type << "(&" << source_nd_name << ")" << emit_bracketed_string(arg0_shape)
           << " = *reinterpret_cast<" << element_type << "(*)" << emit_bracketed_string(arg0_shape)
           << ">(" << arg0 << ");\n";
    writer << element_type << "(&" << dest_nd_name << ")" << emit_bracketed_string(out_shape)
           << " = *reinterpret_cast<" << element_type << "(*)" << emit_bracketed_string(out_shape)
           << ">(" << out << ");\n";
    if (out_shape.size() == 0)
    {
        writer << dest_nd_name << " = 0;\n";
    }
    else
    {
        std::vector<std::string> output_vars;
        for (size_t i = 0; i < out_shape.size(); i++)
        {
            std::string index_var = writer.generate_temporary_name("i");

            writer << start_index_loop(index_var, 0, out_shape[i], i == 0);
            writer.indent++;

            output_vars.push_back(index_var);
        }

        writer << dest_nd_name << emit_bracketed_string(output_vars) << " = 0;\n";

        for (size_t i = out_shape.size(); i-- > 0;)
        {
            writer.indent--;
            writer << end_index_loop(output_vars[i]);
        }
    }
    if (std::find(arg0_shape.begin(), arg0_shape.end(), 0) == arg0_shape.end())
    {
        std::vector<std::string> index_vars;
        for (size_t i = 0; i < arg0_shape.size(); i++)
        {
            std::string index_var = writer.generate_temporary_name("i");
            index_vars.push_back(index_var);
        }
        std::vector<std::string> out_indexes;
        size_t outer_arg_index = -1;
        for (size_t i = 0; i < index_vars.size(); ++i)
        {
            if (reduction_axes.count(i) == 0)
            {
                if (out_indexes.size() == 0)
                {
                    outer_arg_index = i;
                }
                out_indexes.push_back(index_vars[i]);
            }
        }

        if (outer_arg_index != -1)
        {
            writer << start_index_loop(
                index_vars[outer_arg_index], 0, arg0_shape[outer_arg_index], true);
            writer.indent++;
        }
        for (size_t i = 0; i < arg0_shape.size(); i++)
        {
            if (i != outer_arg_index)
            {
                std::string index_var = index_vars[i];
                writer << start_index_loop(index_var, 0, arg0_shape[i], false);
                writer.indent++;
            }
        }

        writer << dest_nd_name << emit_bracketed_string(out_indexes) << " += " << source_nd_name
               << emit_bracketed_string(index_vars) << ";\n";

        for (size_t i = arg0_shape.size(); i-- > 0;)
        {
            writer.indent--;
            writer << end_index_loop(index_vars[i]);
        }
    }
}
