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

#include <CPP/concatenation.hpp>
#include <CPP/custom_gpu_primitive.hpp>
#include <CPP/reshape.hpp>

#include "ngraph/runtime/intelgpu/intelgpu_layout.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_custom_kernels.hpp"

#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

string runtime::intelgpu::get_opencl_type_name(const element::Type& ngraph_type)
{
    switch (ngraph_type.get_type_enum())
    {
    case element::Type_t::i64: return "long";
    case element::Type_t::i32: return "int";
    case element::Type_t::i16: return "short";
    case element::Type_t::u16: return "ushort";
    case element::Type_t::i8: return "char";
    case element::Type_t::u8: return "uchar";
    }

    return ngraph_type.c_type_string();
}

string runtime::intelgpu::get_opencl_type_min_max_value(const element::Type& ngraph_type,
                                                        bool is_min)
{
    switch (ngraph_type.get_type_enum())
    {
    case element::Type_t::f32: return is_min ? "-INFINITY" : "INFINITY";
    case element::Type_t::f64: return is_min ? "-INFINITY" : "INFINITY";
    case element::Type_t::i64: return is_min ? "LONG_MIN" : "LONG_MAX";
    case element::Type_t::u64: return is_min ? "0" : "ULONG_MAX";
    case element::Type_t::i32: return is_min ? "INT_MIN" : "INT_MAX";
    case element::Type_t::u32: return is_min ? "0" : "UINT_MAX";
    case element::Type_t::i16: return is_min ? "SHRT_MIN" : "SHRT_MAX";
    case element::Type_t::u16: return is_min ? "0" : "USHRT_MAX";
    case element::Type_t::i8: return is_min ? "CHAR_MIN" : "CHAR_MAX";
    case element::Type_t::u8: return is_min ? "0" : "UCHAR_MAX";
    }

    throw ngraph_error("Unsupported type '" + ngraph_type.c_type_string() +
                       "' in runtime::intelgpu::get_opencl_type_min_max_value()");
}

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

string runtime::intelgpu::access_dims(const Shape& dimentions,
                                      const string& var,
                                      const AxisSet& axis,
                                      bool is_reversed)
{
    size_t var_idx = 0;
    stringstream buffer;

    for (auto const& i : dimentions)
    {
        if (axis.find(var_idx) == axis.end())
        {
            buffer << "[" << var << var_idx << "]";
        }
        else if (is_reversed)
        {
            buffer << "[" << i << " - " << var << var_idx << " - 1]";
        }
        ++var_idx;
    }

    if (!buffer.rdbuf()->in_avail())
    { // it means scalar
        buffer.str("[0]");
    }

    return buffer.str();
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

vector<size_t> runtime::intelgpu::generate_loops_w_axes(codegen::CodeWriter& writer,
                                                        const Shape& shape,
                                                        bool is_begin,
                                                        const AxisSet& axis,
                                                        const string& expression)
{
    const size_t cldnn_gws_lim = 3;
    vector<size_t> gws;
    size_t var_idx = 0;
    size_t dim_idx = 0;

    if (is_begin)
    {
        for (auto const& i : shape)
        {
            if (axis.find(var_idx) == axis.end())
            {
                if (dim_idx < cldnn_gws_lim)
                {
                    writer << "const unsigned i" << var_idx << " = get_global_id(" << dim_idx
                           << "); /* trip count " << i << "*/\n";
                    gws.push_back(i);
                    ++dim_idx;
                }
                else
                {
                    writer << "for (uint i" << var_idx << " = 0; i" << var_idx << " < " << i
                           << "; ++i" << var_idx << ")\n";
                    writer.block_begin();
                }
            }
            ++var_idx;
        }

        if (!expression.empty())
        {
            writer << expression;
        }

        var_idx = 0;
        for (auto const& i : shape)
        {
            if (axis.find(var_idx) != axis.end())
            {
                writer << "for (uint i" << var_idx << " = 0; i" << var_idx << " < " << i << "; ++i"
                       << var_idx << ")\n";
                writer.block_begin();
            }
            ++var_idx;
        }
    }
    else
    { // is_begin == false
        for (auto const& i : shape)
        {
            if (axis.find(var_idx) != axis.end())
            {
                writer.block_end();
            }
            ++var_idx;
        }

        if (!expression.empty())
        {
            writer << expression;
        }

        var_idx = 0;
        for (auto const& i : shape)
        {
            if (axis.find(var_idx) == axis.end())
            {
                if (dim_idx < cldnn_gws_lim)
                {
                    ++dim_idx;
                }
                else
                {
                    writer.block_end();
                }
            }
            ++var_idx;
        }
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
    const string entry_point_name = "op_pad_" + output_name;
    const size_t cldnn_gws_lim = 3;
    codegen::CodeWriter writer;
    vector<size_t> gws;

    // The kernel name and parameters
    gen_func_def(writer, entry_point_name, {2, "float"}, {input_shape, {1}}, "float", output_shape);

    writer.block_begin();
    {
        // Loop for Copy input matrix into output matrix with padding.
        // Padding include "pad_below" and "pad_interior" according nGraph documentation
        size_t var_idx = 0;
        for (auto const& i : output_shape)
        {
            if (var_idx < cldnn_gws_lim)
            {
                writer << "\nconst uint i" << var_idx << " = get_global_id(" << var_idx
                       << "); /*trip count " << i << "*/\n";
                gws.push_back(i);
            }
            else
            {
                writer << "for (uint i" << var_idx << " = 0; i" << var_idx << " < " << i << "; ++i"
                       << var_idx << ")\n";
            }
            writer.block_begin();

            writer << "uint input_idx" << var_idx << " = i" << var_idx << " - "
                   << pad_below.at(var_idx) << " /*pad_below*/;\n";
            writer << "uint input_idx_interior" << var_idx << " = input_idx" << var_idx << " / ("
                   << pad_interior.at(var_idx) << " /*pad_interior*/ + 1);\n";

            ++var_idx;
        }

        // Generate padding conditionals
        writer << "\n// Since we use unsigned indexes we don't need "
               << "(input_idxX >= 0) extra check\n"
               << "if (";
        var_idx = 0;
        for (auto const& i : input_shape)
        {
            if (var_idx)
            {
                writer << " && ";
            }

            writer << "(input_idx_interior" << var_idx << " < " << i << ") && ((input_idx"
                   << var_idx << " % (" << pad_interior.at(var_idx) << " + 1)) == 0)";

            ++var_idx;
        }
        writer << ")\n";
        writer.block_begin();
        {
            writer << "output" << access_dims(output_shape) << " = input0"
                   << access_dims(input_shape, "input_idx_interior") << ";\n";
        }
        writer.block_end();
        writer << "else\n";
        writer.block_begin();
        {
            writer << "output" << access_dims(output_shape) << " = input1[0];\n";
        } // End of padding conditionals
        writer.block_end();

        // Closing brackets for main Copy loop
        for (auto const& i : output_shape)
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

static void gen_window_loop(codegen::CodeWriter& writer,
                            const Shape& output_shape,
                            const Shape& win_shape,
                            const Shape& win_stride,
                            const Shape& pad_below,
                            bool is_begin)
{
    size_t var_idx = 0;

    if (is_begin)
    {
        for (auto const& i : win_shape)
        {
            writer << "for (uint w" << var_idx << " = 0; w" << var_idx << " < " << i << "; ++w"
                   << var_idx << ")\n";
            writer.block_begin();
            writer << "const uint win_idx" << var_idx << " = (i" << var_idx + 2 << " * "
                   << win_stride.at(var_idx) << " /*win_stride*/)"
                   << " + w" << var_idx << " - " << pad_below.at(var_idx) << " /*pad_below*/;\n";
            ++var_idx;
        }

        writer << "if (";
        // Generate input coordinate condition
        for (size_t i = 0; i < win_shape.size(); ++i)
        {
            if (i)
            {
                writer << " && ";
            }
            writer << "(win_idx" << i << " < " << output_shape.at(i + 2) << ")";
        }
        writer << ")\n";
        writer.block_begin();
    }
    else
    {
        writer.block_end();
        for (auto const& i : win_shape)
        {
            writer.block_end();
        }
    }
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
    gen_func_def(
        writer, entry_point_name, {2, "float"}, {input_shape, delta_shape}, "float", output_shape);

    writer.block_begin();
    {
        // Main loop over delta input array
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
                // input coordinate condition
                gen_window_loop(writer, output_shape, win_shape, win_stride, pad_below, true);

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
                }
                // End of input coordinate condition
                // Closing brackets for window shape loop
                gen_window_loop(writer, output_shape, win_shape, win_stride, pad_below, false);

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

void runtime::intelgpu::do_avg_pool_backprop_operation(cldnn::topology& topology,
                                                       const string& delta_name,
                                                       const Shape& delta_shape,
                                                       const string& output_name,
                                                       const Shape& output_shape,
                                                       const element::Type& output_type,
                                                       const Shape& win_shape,
                                                       const Shape& win_stride,
                                                       const Shape& pad_below,
                                                       const bool include_padding)
{
    const string entry_point_name = "op_avg_pool_backprop_" + output_name;
    codegen::CodeWriter writer;
    vector<size_t> gws;

    const Shape delta_data(delta_shape.cbegin() + 2, delta_shape.cend());
    const Shape output_data(output_shape.cbegin() + 2, output_shape.cend());

    size_t win_elems_size = shape_size<Shape>(win_shape);

    // The kernel name and parameters
    gen_func_def(writer, entry_point_name, {"float"}, {delta_shape}, "float", output_shape);

    writer.block_begin();
    {
        writer << "size_t win_elems_size = " << win_elems_size << ";\n";
        writer << "float computed_val = 0.0f;\n";

        // Main loop over delta input array
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

                if (!include_padding)
                {
                    writer << "win_elems_size = 0;\n";

                    // Loop over window shape
                    // input coordinate condition
                    gen_window_loop(writer, output_shape, win_shape, win_stride, pad_below, true);

                    writer << "++win_elems_size;\n";

                    // End of input coordinate condition
                    // Closing brackets for window shape loop
                    gen_window_loop(writer, output_shape, win_shape, win_stride, pad_below, false);
                }

                // Loop over window shape
                // input coordinate condition
                gen_window_loop(writer, output_shape, win_shape, win_stride, pad_below, true);

                writer << "computed_val = input0" << access_dims(delta_shape)
                       << " / win_elems_size;\n";

                writer << "output[i0][i1]";
                // additional dimensions for input
                for (size_t i = 0; i < win_shape.size(); ++i)
                {
                    writer << "[win_idx" << i << "]";
                }
                writer << " += computed_val;\n";

                // End of input coordinate condition
                // Closing brackets for window shape loop
                gen_window_loop(writer, output_shape, win_shape, win_stride, pad_below, false);

                // Closing brackets for delta loop
                for (auto const& i : delta_data)
                {
                    writer.block_end();
                }
            }
            // End of loop over i1
            writer.block_end();
        }
        // End of loop over i0
        writer.block_end();
    }
    // End of function bracket
    writer.block_end();

    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const cldnn::custom_gpu_primitive op_avg_pool_backprop(output_name,
                                                           {delta_name},
                                                           {writer.get_code()},
                                                           entry_point_name,
                                                           get_kernel_args(1, 1),
                                                           "",
                                                           layout,
                                                           gws);
    topology.add(op_avg_pool_backprop);
}

void runtime::intelgpu::do_dot_operation(cldnn::topology& topology,
                                         const string& input0_name,
                                         const Shape& input0_shape,
                                         const string& input1_name,
                                         const Shape& input1_shape,
                                         const string& output_name,
                                         const Shape& output_shape,
                                         const element::Type& output_type,
                                         size_t reduction_axes_count)
{
    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    string entry_point_name = "dot_" + output_name;
    const string type_name = get_opencl_type_name(output_type);
    const size_t input0_axes = input0_shape.size() - reduction_axes_count;
    size_t var_idx = reduction_axes_count;
    Shape reduction_shape;
    codegen::CodeWriter writer;
    vector<size_t> gws;

    for (auto it = input1_shape.begin(); (it != input1_shape.end()) && (var_idx > 0); ++it)
    {
        reduction_shape.push_back(*it);
        --var_idx;
    }

    runtime::intelgpu::gen_func_def(writer,
                                    entry_point_name,
                                    {2, type_name},
                                    {input0_shape, input1_shape},
                                    type_name,
                                    output_shape);
    writer.block_begin();
    {
        writer << "// reduction_axes_count:" << reduction_axes_count << "\n"
               << "// reduction_shape:" << reduction_shape << "\n";

        // Main loops
        gws = runtime::intelgpu::generate_loops(writer, output_shape, true);

        writer << type_name << " sum = 0;\n";

        // Reduction loops
        var_idx = 0;
        for (auto const& i : reduction_shape)
        {
            writer << "for (uint k" << var_idx << " = 0; k" << var_idx << " < " << i << "; ++k"
                   << var_idx << ")\n";
            writer.block_begin();

            ++var_idx;
        }

        writer << "sum += input0";

        if (input0_shape.empty())
        {
            writer << "[0]";
        }
        else
        {
            // main axes indexes
            for (size_t i = 0; i < input0_axes; ++i)
            {
                writer << "[i" << i << "]";
            }

            // reduction axes indexes
            for (size_t i = 0; i < reduction_shape.size(); ++i)
            {
                writer << "[k" << i << "]";
            }
        }

        // operation
        writer << " * input1";

        if (input1_shape.empty())
        {
            writer << "[0]";
        }
        else
        {
            // reduction axes indexes
            for (size_t i = 0; i < reduction_shape.size(); ++i)
            {
                writer << "[k" << i << "]";
            }

            // main axes indexes
            for (size_t i = input0_axes; i < output_shape.size(); ++i)
            {
                writer << "[i" << i << "]";
            }
        }

        writer << ";\n";

        //  Closing brackets for reduction loops
        for (auto const& i : reduction_shape)
        {
            writer.block_end();
        }

        writer << "output" << runtime::intelgpu::access_dims(output_shape) << " = sum;\n";

        // Closing brackets for main loops
        runtime::intelgpu::generate_loops(writer, output_shape, false);
    }
    writer.block_end();

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

    gen_func_def(writer,
                 entry_point_name,
                 {get_opencl_type_name(output_type)},
                 {input_shape},
                 get_opencl_type_name(output_type),
                 output_shape);

    writer.block_begin();
    {
        // Main loops
        gws = generate_loops(writer, output_shape, true);

        writer << "output" << access_dims(output_shape) << " = input0"
               << access_dims_strided(input_shape, lower_bounds, strides, false) << ";\n";

        // Closing brackets for main loops
        generate_loops(writer, output_shape, false);
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

    gen_func_def(writer,
                 entry_point_name,
                 {"char", "float", "float"},
                 {input0_shape, input1_shape, input2_shape},
                 "float",
                 output_shape);

    writer.block_begin();
    {
        // Main loops
        gws = generate_loops(writer, output_shape, true);

        writer << "output" << access_dims(output_shape) << " = input0" << access_dims(input0_shape)
               << " ? input1" << access_dims(input1_shape) << " : input2"
               << access_dims(input2_shape) << ";\n";

        // Closing brackets for main loops
        generate_loops(writer, output_shape, false);
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
                                        const element::Type& input0_type,
                                        const string& input1_name,
                                        const Shape& input1_shape,
                                        const string& output_name,
                                        const Shape& output_shape,
                                        const element::Type& output_type,
                                        const string& operation)
{
    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const string entry_point_name = "logic_" + output_name;
    codegen::CodeWriter writer;
    vector<size_t> gws;

    gen_func_def(writer,
                 entry_point_name,
                 {2, get_opencl_type_name(input0_type)},
                 {input0_shape, input1_shape},
                 get_opencl_type_name(output_type),
                 output_shape);

    writer.block_begin();
    {
        // Main loops
        gws = generate_loops(writer, output_shape, true);

        writer << "output" << access_dims(output_shape) << " = input0" << access_dims(input0_shape)
               << operation << "input1" << access_dims(input1_shape) << " ? 1 : 0;\n";

        // Closing brackets for main loops
        generate_loops(writer, output_shape, false);
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

void runtime::intelgpu::do_eltwise_kernel(cldnn::topology& topology,
                                          const string& input0_name,
                                          const Shape& input0_shape,
                                          const element::Type& input0_type,
                                          const string& input1_name,
                                          const Shape& input1_shape,
                                          const string& output_name,
                                          const Shape& output_shape,
                                          const element::Type& output_type,
                                          const string& operation)
{
    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const string entry_point_name = "eltwise_" + output_name;
    codegen::CodeWriter writer;
    vector<size_t> gws;

    gen_func_def(writer,
                 entry_point_name,
                 {2, get_opencl_type_name(input0_type)},
                 {input0_shape, input1_shape},
                 get_opencl_type_name(output_type),
                 output_shape);

    writer.block_begin();
    {
        // Main loops
        gws = generate_loops(writer, output_shape, true);

        writer << "output" << access_dims(output_shape) << " = " << operation << "(input0"
               << access_dims(input0_shape) << ", input1" << access_dims(input1_shape) << ");\n";

        // Closing brackets for main loops
        generate_loops(writer, output_shape, false);
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

    gen_func_def(writer, entry_point_name, {"float"}, {input_shape}, "float", output_shape);

    writer.block_begin();
    {
        gws = generate_loops(writer, output_shape, true);

        writer << "output" << access_dims(output_shape) << " = input0"
               << access_dims(output_shape, "i", reversed_axes, true) << ";\n";

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

    gen_func_def(writer, entry_point_name, {"char"}, {input_shape}, "char", output_shape);

    writer.block_begin();
    {
        gws = generate_loops(writer, output_shape, true);

        writer << "output" << access_dims(output_shape) << " = !input0" << access_dims(input_shape)
               << ";\n";

        generate_loops(writer, output_shape, false);
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

    gen_func_def(writer,
                 entry_point_name,
                 {get_opencl_type_name(input_type)},
                 {input_shape},
                 get_opencl_type_name(output_type),
                 output_shape);

    writer.block_begin();
    {
        writer << "for (uint i = 0; i < " << output_shape.at(one_hot_axis) << "; ++i)\n";
        writer.block_begin();
        {
            gws = generate_loops(writer, input_shape, true);

            size_t current_input = 0;
            string buffer;
            const size_t output_shape_size = output_shape.size();
            for (uint j = 0; j < output_shape_size; ++j)
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

            generate_loops(writer, input_shape, false);
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
    const string& input_type_name = get_opencl_type_name(input_type);
    const string& output_type_name = get_opencl_type_name(output_type);
    codegen::CodeWriter writer;
    vector<size_t> gws;

    gen_func_def(
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

void runtime::intelgpu::do_sigmoid_backprop_operation(cldnn::topology& topology,
                                                      const string& input_name,
                                                      const Shape& input_shape,
                                                      const string& delta_name,
                                                      const Shape& delta_shape,
                                                      const string& output_name,
                                                      const Shape& output_shape,
                                                      const element::Type& output_type)
{
    const string entry_point_name = "op_sigmoid_backprop_" + output_name;
    codegen::CodeWriter writer;
    vector<size_t> gws;

    gen_func_def(
        writer, entry_point_name, {2, "float"}, {input_shape, delta_shape}, "float", output_shape);

    writer.block_begin();
    {
        writer << "float func_x = 0.0f;\n";
        gws = generate_loops(writer, output_shape, true);

        writer << "func_x = 1.0f/(1.0f+ exp(-input0" << access_dims(input_shape) << "));\n";
        writer << "output" << access_dims(output_shape) << " = input1" << access_dims(delta_shape)
               << " * func_x * (1.0f - func_x);\n";

        generate_loops(writer, output_shape, false);
    }
    writer.block_end();

    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const cldnn::custom_gpu_primitive op_sigmoid_backprop(output_name,
                                                          {input_name, delta_name},
                                                          {writer.get_code()},
                                                          entry_point_name,
                                                          get_kernel_args(2, 1),
                                                          "",
                                                          layout,
                                                          gws);
    topology.add(op_sigmoid_backprop);
}

void runtime::intelgpu::do_custom_eltwise_operation(cldnn::topology& topology,
                                                    const string& input_name,
                                                    const Shape& input_shape,
                                                    const element::Type& input_type,
                                                    const string& output_name,
                                                    const Shape& output_shape,
                                                    const element::Type& output_type,
                                                    const CUSTOM_ELTWISE operation_name)
{
    const string entry_point_name = "op_custom_eltwise_" + output_name;
    codegen::CodeWriter writer;
    vector<size_t> gws;

    gen_func_def(writer,
                 entry_point_name,
                 {get_opencl_type_name(input_type)},
                 {input_shape},
                 get_opencl_type_name(output_type),
                 output_shape);
    writer.block_begin();
    {
        gws = generate_loops(writer, output_shape, true);
        writer << "output" << access_dims(output_shape) << " = ";
        switch (operation_name)
        {
        case CUSTOM_ELTWISE::Atan:
        {
            writer << "atan";
            break;
        }
        case CUSTOM_ELTWISE::Ceil:
        {
            writer << "ceil";
            break;
        }
        case CUSTOM_ELTWISE::Floor:
        {
            if (input_type.is_real())
            {
                writer << "floor";
            }
            break;
        }
        case CUSTOM_ELTWISE::Sign:
        {
            writer << "sign";
            break;
        }
        case CUSTOM_ELTWISE::Tan:
        {
            writer << "tan";
            break;
        }
        }
        writer << "(input0" << access_dims(input_shape) << ");\n";
        generate_loops(writer, output_shape, false);
    }
    writer.block_end();

    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const cldnn::custom_gpu_primitive op_custom_eltwise(output_name,
                                                        {input_name},
                                                        {writer.get_code()},
                                                        entry_point_name,
                                                        get_kernel_args(1, 1),
                                                        "",
                                                        layout,
                                                        gws);
    topology.add(op_custom_eltwise);
}

void runtime::intelgpu::do_arg_max_min_operation(cldnn::topology& topology,
                                                 const string& input_name,
                                                 const Shape& input_shape,
                                                 const element::Type& input_type,
                                                 const string& output_name,
                                                 const Shape& output_shape,
                                                 const element::Type& output_type,
                                                 const size_t reduction_axis,
                                                 const bool is_max)
{
    const string operation_name = is_max ? "max" : "min";
    const string entry_point_name = "op_arg_" + operation_name + "_" + output_name;
    codegen::CodeWriter writer;
    vector<size_t> gws;

    const string operation_sign = is_max ? " > " : " < ";
    const string infinity = get_opencl_type_min_max_value(input_type, is_max);
    const string var_name = operation_name + "_val";

    size_t current_input = 0;
    string dims_buffer;
    const size_t input_shape_size = input_shape.size();
    for (uint j = 0; j < input_shape_size; ++j)
    {
        if (j == reduction_axis)
        {
            dims_buffer += "[i]";
        }
        else
        {
            dims_buffer += "[i" + to_string(current_input) + "]";
            ++current_input;
        }
    }

    gen_func_def(writer,
                 entry_point_name,
                 {get_opencl_type_name(input_type)},
                 {input_shape},
                 get_opencl_type_name(output_type),
                 output_shape);

    writer.block_begin();
    {
        gws = generate_loops(writer, output_shape, true);

        writer << get_opencl_type_name(input_type) << " " << var_name << " = " << infinity << ";\n";
        writer << get_opencl_type_name(output_type) << " index = 0;\n";

        writer << "for (uint i = 0; i < " << input_shape.at(reduction_axis) << "; ++i)\n";
        writer.block_begin();
        {
            writer << "if (input0" << dims_buffer << operation_sign << var_name << ")\n";
            writer.block_begin();
            {
                writer << var_name << " = input0" << dims_buffer << ";\n";
                writer << "index = i;\n";
            }
            writer.block_end();
        }
        writer.block_end();

        writer << "output" << access_dims(output_shape) << " = index;\n";

        generate_loops(writer, output_shape, false);
    }
    writer.block_end();

    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const cldnn::custom_gpu_primitive op_arg_max_min(output_name,
                                                     {input_name},
                                                     {writer.get_code()},
                                                     entry_point_name,
                                                     get_kernel_args(1, 1),
                                                     "",
                                                     layout,
                                                     gws);
    topology.add(op_arg_max_min);
}

void runtime::intelgpu::do_negative_operation(cldnn::topology& topology,
                                              const string& input_name,
                                              const Shape& input_shape,
                                              const element::Type& input_type,
                                              const string& output_name,
                                              const Shape& output_shape,
                                              const element::Type& output_type)
{
    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const string entry_point_name = "negative_" + output_name;
    const string& input_type_name = get_opencl_type_name(input_type);
    const string& output_type_name = get_opencl_type_name(output_type);
    codegen::CodeWriter writer;
    vector<size_t> gws;

    gen_func_def(
        writer, entry_point_name, {input_type_name}, {input_shape}, output_type_name, output_shape);

    writer.block_begin();
    {
        gws = generate_loops(writer, output_shape, true);

        writer << "output" << access_dims(output_shape) << " = - (input0"
               << access_dims(input_shape) << ");\n";

        generate_loops(writer, output_shape, false);
    }
    writer.block_end();

    const cldnn::custom_gpu_primitive op_negative(output_name,
                                                  {input_name},
                                                  {writer.get_code()},
                                                  entry_point_name,
                                                  get_kernel_args(1, 1),
                                                  "",
                                                  layout,
                                                  gws);
    topology.add(op_negative);
}
