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
#include <map>

#include "ngraph/codegen/code_writer.hpp"
#include "ngraph/runtime/gpu/gpu_kernel_emitters.hpp"

using namespace ngraph;
using namespace ngraph::runtime::gpu::kernel;

// Function to take a vector of data, say 1,2,3 and return
// a string representing multi-index access, i.e "[1][2][3]"
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

// Convert a buffer into a C-style multi-index array
std::string recast_tmp_var(codegen::CodeWriter& writer,
                           const std::string& element_type,
                           const std::string& arg_name,
                           const Shape& arg_shape,
                           const std::string& tmp_name)
{
    std::string nd_name = writer.generate_temporary_name(tmp_name);
    std::string bracketed_shape = emit_bracketed_string(arg_shape);

    writer << element_type << "(&" << nd_name << ")" << bracketed_shape << " = *reinterpret_cast<"
           << element_type << "(*)" << bracketed_shape << ">(" << arg_name << ");\n";
    return nd_name;
}

// write openings to for loops, for variables in the order of top,
// where each loop ranges from bottom[i] to top[i]
// creates index variables for each loop and returns them
std::vector<std::string>
    open_for_loops(codegen::CodeWriter& writer, const Shape& top, const Shape& bottom = {})
{
    Shape new_bottom;
    if (bottom.size() == 0)
    {
        new_bottom = Shape(top.size(), 0);
    }
    else
    {
        new_bottom = bottom;
    }

    std::vector<std::string> index_vars;
    for (size_t i = 0; i < top.size(); i++)
    {
        std::string index_var = writer.generate_temporary_name("i");

        // writer << start_index_loop(index_var, new_bottom[i], top[i], i == 0);
        writer.indent++;

        index_vars.push_back(index_var);
    }

    return index_vars;
}
//close the for loops created by open_for_loops
void close_for_loops(codegen::CodeWriter& writer, const std::vector<std::string>& index_vars)
{
    for (size_t i = index_vars.size(); i-- > 0;)
    {
        writer.indent--;
        // writer << end_index_loop(index_vars[i]);
    }
}

void ngraph::runtime::gpu::kernel::emit_broadcast(codegen::CodeWriter& writer,
                                                   const std::string& element_type,
                                                   const std::string& arg0, // replacement context
                                                   const std::string& out,
                                                   const Shape& arg0_shape,
                                                   const Shape& out_shape,
                                                   const AxisSet& broadcast_axes)
{
    // create input and output arrays
    auto source_nd_name = recast_tmp_var(writer, element_type, arg0, arg0_shape, "source_nd");
    auto dest_nd_name = recast_tmp_var(writer, element_type, out, out_shape, "dest_nd");

    // create the for loops
    auto index_vars = open_for_loops(writer, out_shape);

    // match positions in output to positions in the input
    std::vector<std::string> source_indexes;
    for (size_t i = 0; i < out_shape.size(); ++i)
    {
        if (broadcast_axes.count(i) == 0)
        {
            source_indexes.push_back(index_vars[i]);
        }
    }
    // write the operation
    writer << dest_nd_name << emit_bracketed_string(index_vars) << " = " << source_nd_name
           << emit_bracketed_string(source_indexes) << ";\n";

    close_for_loops(writer, index_vars);
}

//
// For the reference kernel this is gpud on, see ngraph/runtime/kernel/concat.hpp.
//
void ngraph::runtime::gpu::kernel::emit_concat(codegen::CodeWriter& writer,
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

void ngraph::runtime::gpu::kernel::emit_replace_slice(
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
    // CoordinateTransform copy_transform(out_shape);
    // emit_pointwise_copy(writer, element_type, arg0, out, copy_transform, copy_transform);

    // // Step 2: Overwrite the slice for replacement.
    // CoordinateTransform input_transform(arg1_shape);
    // CoordinateTransform output_transform(out_shape, lower_bounds, upper_bounds, strides);

    // emit_pointwise_copy(writer, element_type, arg1, out, input_transform, output_transform);
}

void ngraph::runtime::gpu::kernel::emit_slice(codegen::CodeWriter& writer,
                                               const std::string& element_type,
                                               const std::string& arg0, // replacement context
                                               const std::string& out,
                                               const Shape& arg0_shape,
                                               const Shape& out_shape,
                                               const Coordinate& lower_bounds,
                                               const Coordinate& upper_bounds,
                                               const Strides& strides)
{
    // create input and output arrays
    // auto source_nd_name = recast_tmp_var(writer, element_type, arg0, arg0_shape, "source_nd");
    // auto dest_nd_name = recast_tmp_var(writer, element_type, out, out_shape, "dest_nd");

    // // create the for loops
    // auto index_vars = open_for_loops(writer, out_shape);

    // // map the position in the output to a position in the input
    // std::vector<std::string> source_indexes;
    // size_t j = 0;
    // for (size_t i = 0; i < lower_bounds.size(); ++i)
    // {
    //     if (lower_bounds[i] == upper_bounds[i])
    //     {
    //         source_indexes.push_back(std::to_string(lower_bounds[i]));
    //     }
    //     else
    //     {
    //         std::stringstream ss;
    //         ss << lower_bounds[i];
    //         ss << " + " << index_vars[j];
    //         ss << " * " << strides[i];
    //         source_indexes.push_back(ss.str());
    //         j += 1;
    //     }
    // }

    // // write the element copy operation
    // writer << dest_nd_name << emit_bracketed_string(index_vars) << " = " << source_nd_name
    //        << emit_bracketed_string(source_indexes) << ";\n";

    // close_for_loops(writer, index_vars);
}

void ngraph::runtime::gpu::kernel::emit_reshape(codegen::CodeWriter& writer,
                                                 const std::string& element_type,
                                                 const std::string& arg0, // replacement context
                                                 const std::string& out,
                                                 const Shape& arg0_shape,
                                                 const Shape& out_shape,
                                                 const AxisVector& arg0_axis_order)
{
    // // get the total number of elements
    // size_t size = 1;
    // for (auto x : out_shape)
    // {
    //     if (x != 0)
    //         size *= x;
    // }

    // // create input and output arrays
    // auto source_nd_name = recast_tmp_var(writer, element_type, arg0, arg0_shape, "source_nd");
    // auto dest_nd_name = recast_tmp_var(writer, element_type, out, {size}, "dest_nd");

    // std::map<size_t, size_t> input_to_loop_pos;
    // std::map<size_t, size_t> loop_to_input_pos;
    // // loop over the input in the order of arg0_axis_order
    // int input_pos = 0;
    // Shape ordered_input_shape;
    // for (size_t i = 0; i < arg0_shape.size(); i++)
    // {
    //     ordered_input_shape.push_back(arg0_shape[arg0_axis_order[i]]);
    //     input_to_loop_pos[input_pos] = arg0_axis_order[i];
    //     input_pos += 1;
    // }

    // for (auto kv : input_to_loop_pos)
    // {
    //     loop_to_input_pos[kv.second] = kv.first;
    // }

    // auto index_vars = open_for_loops(writer, ordered_input_shape);

    // // write the output reshape as a 1D array by calculating the
    // // position of the input iterators in the output array
    // writer << dest_nd_name << "[ 0";

    // for (size_t i = 0; i < arg0_shape.size(); i++)
    // {
    //     writer << " + " << index_vars[i];
    //     for (auto j = i + 1; j < arg0_shape.size(); j++)
    //     {
    //         if (arg0_shape[j] > 0)
    //         {
    //             writer << " * " << ordered_input_shape[j];
    //         }
    //     }
    // }
    // writer << "] = " << source_nd_name;

    // for (size_t i = 0; i < arg0_shape.size(); i++)
    // {
    //     writer << "[" << index_vars[loop_to_input_pos[i]] << "]";
    // }
    // writer << ";\n";

    // close_for_loops(writer, index_vars);
}

void ngraph::runtime::gpu::kernel::emit_sum(codegen::CodeWriter& writer,
                                             const std::string& element_type,
                                             const std::string& arg0, // replacement context
                                             const std::string& out,
                                             const Shape& arg0_shape,
                                             const Shape& out_shape,
                                             const AxisSet& reduction_axes)
{
    // // create input and output arrays
    // auto source_nd_name = recast_tmp_var(writer, element_type, arg0, arg0_shape, "source_nd");
    // auto dest_nd_name = recast_tmp_var(writer, element_type, out, out_shape, "dest_nd");

    // // zero the output to make sure we don't have randomly initialized data
    // if (out_shape.size() == 0)
    // {
    //     writer << dest_nd_name << " = 0;\n";
    // }
    // else
    // {
    //     auto output_vars = open_for_loops(writer, out_shape);

    //     writer << dest_nd_name << emit_bracketed_string(output_vars) << " = 0;\n";

    //     close_for_loops(writer, output_vars);
    // }

    // // If we don't have a zero index in the input, perform the sum
    // if (std::find(arg0_shape.begin(), arg0_shape.end(), 0) == arg0_shape.end())
    // {
    //     // create the the interation variables without writing the for loops
    //     std::vector<std::string> index_vars;
    //     for (size_t i = 0; i < arg0_shape.size(); i++)
    //     {
    //         std::string index_var = writer.generate_temporary_name("i");
    //         index_vars.push_back(index_var);
    //     }

    //     // calculate the output indexes gpud on what's being reduced
    //     std::vector<std::string> out_indexes;
    //     size_t outer_arg_index = -1;
    //     for (size_t i = 0; i < index_vars.size(); ++i)
    //     {
    //         if (reduction_axes.count(i) == 0)
    //         {
    //             if (out_indexes.size() == 0)
    //             {
    //                 outer_arg_index = i;
    //             }
    //             out_indexes.push_back(index_vars[i]);
    //         }
    //     }

    //     // make the first output shape our outer loop, optimize with openmp
    //     if (outer_arg_index != -1)
    //     {
    //         writer << start_index_loop(
    //             index_vars[outer_arg_index], 0, arg0_shape[outer_arg_index], true);
    //         writer.indent++;
    //     }

    //     // create the rest of the loops, don't parallelize.
    //     for (size_t i = 0; i < arg0_shape.size(); i++)
    //     {
    //         if (i != outer_arg_index)
    //         {
    //             std::string index_var = index_vars[i];
    //             writer << start_index_loop(index_var, 0, arg0_shape[i], false);
    //             writer.indent++;
    //         }
    //     }

    //     writer << dest_nd_name << emit_bracketed_string(out_indexes) << " += " << source_nd_name
    //            << emit_bracketed_string(index_vars) << ";\n";

    //     close_for_loops(writer, index_vars);
    // }
}
