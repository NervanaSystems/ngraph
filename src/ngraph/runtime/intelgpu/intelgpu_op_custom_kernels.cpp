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

#include <sys/resource.h>
#include <sys/time.h>

#include <CPP/concatenation.hpp>
#include <CPP/custom_gpu_primitive.hpp>
#include <CPP/reshape.hpp>

#include "ngraph/runtime/intelgpu/intelgpu_kernels.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_layout.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_custom_kernels.hpp"

#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;
using namespace ngraph::runtime::intelgpu;

string runtime::intelgpu::get_opencl_type_name(const element::Type& ngraph_type)
{
    switch (ngraph_type.get_type_enum())
    {
    case element::Type_t::i64: return "long";
    case element::Type_t::u64: return "ulong";
    case element::Type_t::i32: return "int";
    case element::Type_t::u32: return "uint";
    case element::Type_t::i16: return "short";
    case element::Type_t::u16: return "ushort";
    case element::Type_t::i8: return "char";
    case element::Type_t::u8: return "uchar";
    case element::Type_t::boolean: return "bool";
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
    case element::Type_t::boolean: return is_min ? "0" : "1";
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

void runtime::intelgpu::gen_func_def(CodeWriter& writer,
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

vector<size_t>
    runtime::intelgpu::generate_loops(CodeWriter& writer, const Shape& shape, bool is_begin)
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

vector<size_t> runtime::intelgpu::generate_loops_w_axes(CodeWriter& writer,
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
                                         const CoordinateDiff& pad_below)
{
    const string entry_point_name = "op_pad_" + output_name;
    const size_t cldnn_gws_lim = 3;
    CodeWriter writer;
    vector<size_t> gws;

    // FIXME: Compatibility hack added by amprocte now that interior padding has been removed
    // from nGraph's Pad op.
    Shape pad_interior(pad_below.size(), 0);

    // The kernel name and parameters
    gen_func_def(writer,
                 entry_point_name,
                 {2, get_opencl_type_name(output_type)},
                 {input_shape, {1}},
                 get_opencl_type_name(output_type),
                 output_shape);

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

static void gen_window_loop(CodeWriter& writer,
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
    const string type_name = get_opencl_type_name(output_type);
    const Shape delta_data(delta_shape.cbegin() + 2, delta_shape.cend());
    const Shape output_data(output_shape.cbegin() + 2, output_shape.cend());
    CodeWriter writer;
    vector<size_t> gws;

    // The kernel name and parameters
    gen_func_def(writer,
                 entry_point_name,
                 {2, type_name},
                 {input_shape, delta_shape},
                 type_name,
                 output_shape);

    writer.block_begin();
    {
        // Main loop over delta input array
        writer << "const uint i0 = get_global_id(0);";
        gws.push_back(delta_shape.at(0));
        writer << "/*trip count " << delta_shape.at(0) << "*/\n";
        writer.block_begin();
        {
            writer << "const uint i1 = get_global_id(1);";
            gws.push_back(delta_shape.at(1));
            writer << "/*trip count " << delta_shape.at(1) << "*/\n";
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
                writer << " = 0;\n";

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
                writer << type_name
                       << " max_elem = " << get_opencl_type_min_max_value(output_type, true)
                       << ";\n"
                       << "uint elem_exists = 0;\n";

                // Loop over window shape
                // input coordinate condition
                gen_window_loop(writer, output_shape, win_shape, win_stride, pad_below, true);

                {
                    writer << "const " << type_name << " max_local = input0[i0][i1]";
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

void runtime::intelgpu::do_max_avg_pool_operation(cldnn::topology& topology,
                                                  const string& input_name,
                                                  const Shape& input_shape,
                                                  const string& output_name,
                                                  const Shape& output_shape,
                                                  const element::Type& output_type,
                                                  const Shape& win_shape,
                                                  const Shape& win_stride,
                                                  const Shape& pad_below,
                                                  bool include_padding,
                                                  const string& def_val,
                                                  bool is_max_pool)
{
    const string entry_point_name = "op_pool_" + to_string(is_max_pool) + "_" + output_name;
    const string type_name = get_opencl_type_name(output_type);
    const string init_accumulator = is_max_pool ? "-FLT_MAX" : def_val;
    CodeWriter writer;
    vector<size_t> gws;

    const Shape input_data(input_shape.cbegin() + 2, input_shape.cend());
    const Shape output_data(output_shape.cbegin() + 2, output_shape.cend());

    // The kernel name and parameters
    gen_func_def(writer, entry_point_name, {type_name}, {input_shape}, type_name, output_shape);

    writer.block_begin();
    { // Main function body

        writer << "//Window:" << win_shape << " Stride: " << win_stride << "\n"
               << "//padding included:" << include_padding << "\n"
               << "//init value:" << def_val << "\n\n";

        writer << "const uint N_dim = get_global_id(0);/*trip count " << input_shape.at(0)
               << "*/\n";
        gws.push_back(output_shape.at(0));
        writer << "const uint C_dim = get_global_id(1);/*trip count " << input_shape.at(1)
               << "*/\n";
        gws.push_back(output_shape.at(1));

        // Loops over output dimensions
        size_t var_idx = 0;
        for (auto i = output_data.begin(); i != output_data.end(); ++i)
        {
            writer << "for (uint i" << var_idx << " = 0; i" << var_idx << " < " << *i << "; ++i"
                   << var_idx << ")\n";
            writer.block_begin();

            ++var_idx;
        }

        writer << type_name << " accumulator = " << init_accumulator << ";\n"
               << "uint element_count = 0;\n\n";

        // Loop over window
        writer << "// Over window iterations\n";

        var_idx = 0;
        for (auto const i : win_shape)
        {
            writer << "for (uint f" << var_idx << " = 0; f" << var_idx << " < " << i << "; ++f"
                   << var_idx << ")\n";
            writer.block_begin();

            writer << "uint input_idx" << var_idx << " = (i" << var_idx << " * "
                   << win_stride.at(var_idx) << " /*win_stride*/"
                   << ") + (f" << var_idx << ")"
                   << " - " << pad_below.at(var_idx) << " /*pad_below*/;\n";
            ++var_idx;
        }

        // Generate conditionals
        writer << "if (";
        var_idx = 0;
        for (auto const& i : input_data)
        {
            if (var_idx)
            {
                writer << " && ";
            }

            writer << "(input_idx" << var_idx << " < " << i << ")";

            ++var_idx;
        }
        writer << ")\n";
        writer.block_begin();
        {
            // Output element calculation
            if (is_max_pool)
            {
                writer << "accumulator = max(accumulator, input0[N_dim][C_dim]"
                       << access_dims(win_shape, "input_idx") << ");\n";
            }
            else
            {
                writer << "accumulator += input0[N_dim][C_dim]"
                       << access_dims(win_shape, "input_idx") << ";\n";
            }
            writer << "++element_count;\n";
        }
        writer.block_end();

        if (include_padding)
        {
            writer << "else\n";
            writer.block_begin();
            {
                // Output element calculation
                writer << "accumulator += " << def_val << ";\n"
                       << "++element_count;\n";
            }
            writer.block_end();
        }

        // End of conditional generation

        // Closing brackets for window loop
        for (auto const& i : win_shape)
        {
            writer.block_end();
        }

        writer << "\nif (element_count)\n";
        writer.block_begin();
        {
            writer << "output[N_dim][C_dim]" << access_dims(output_data) << " = accumulator";
            if (!is_max_pool)
            {
                writer << " / element_count";
            }
            writer << ";\n";
        }
        writer.block_end();

        writer << "else\n";
        writer.block_begin();
        {
            writer << "output[N_dim][C_dim]" << access_dims(output_data) << " = "
                   << init_accumulator << ";\n";
        }
        writer.block_end();

        // Closing brackets for output dimensions
        for (const auto i : output_data)
        {
            writer.block_end();
        }

    } // Main function body
    writer.block_end();

    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const cldnn::custom_gpu_primitive op_avg_pool(output_name,
                                                  {input_name},
                                                  {writer.get_code()},
                                                  entry_point_name,
                                                  get_kernel_args(1, 1),
                                                  "",
                                                  layout,
                                                  gws);
    topology.add(op_avg_pool);
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
    const string type_name = get_opencl_type_name(output_type);
    CodeWriter writer;
    vector<size_t> gws;

    const Shape delta_data(delta_shape.cbegin() + 2, delta_shape.cend());
    const Shape output_data(output_shape.cbegin() + 2, output_shape.cend());

    size_t win_elems_size = shape_size<Shape>(win_shape);

    // The kernel name and parameters
    gen_func_def(writer, entry_point_name, {type_name}, {delta_shape}, type_name, output_shape);

    writer.block_begin();
    {
        writer << "size_t win_elems_size = " << win_elems_size << ";\n";
        writer << type_name << " computed_val = 0.0;\n";

        // Main loop over delta input array
        writer << "const uint i0 = get_global_id(0);";
        gws.push_back(delta_shape.at(0));
        writer << "/*trip count " << delta_shape.at(0) << "*/\n";
        writer.block_begin();
        {
            writer << "const uint i1 = get_global_id(1);";
            gws.push_back(delta_shape.at(1));
            writer << "/*trip count " << delta_shape.at(1) << "*/\n";
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
                writer << " = 0;\n";

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
    CodeWriter writer;
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

CustomKernels::krnl_info CustomKernels::build_krnl(const shared_ptr<op::Gemm>& op) const
{
    const string& input0_name = op->get_input_tensor_name(0);
    const Shape& input0_shape = op->get_input_shape(0);
    const string& input1_name = op->get_input_tensor_name(1);
    const Shape& input1_shape = op->get_input_shape(1);
    const string& input2_name = op->get_input_tensor_name(2);
    const Shape& input2_shape = op->get_input_shape(2);
    const string& output_name = op->get_output_tensor_name(0);
    const Shape& output_shape = op->get_output_shape(0);
    const element::Type& output_type = op->get_output_element_type(0);
    const double alpha = op->get_alpha();
    const double beta = op->get_beta();
    const bool transA = op->get_transA();
    const bool transB = op->get_transB();

    string entry_point_name = "gemm_" + output_name;
    const string type_name = get_opencl_type_name(output_type);
    CodeWriter writer;
    vector<size_t> gws;

    gen_func_def(writer,
                 entry_point_name,
                 {3, type_name},
                 {input0_shape, input1_shape, input2_shape},
                 type_name,
                 output_shape);
    writer.block_begin();
    {
        writer << type_name << " temp[" << output_shape.at(0) << "][" << output_shape.at(1)
               << "];\n";

        writer << "for(uint i0 = 0; i0 < " << output_shape.at(0) << "; ++i0)\n";
        writer.block_begin();
        {
            writer << "for(uint i1 = 0; i1 < " << output_shape.at(1) << "; ++i1)\n";
            writer.block_begin();
            {
                string input2_coords;
                if (input2_shape.empty())
                {
                    input2_coords = "[0]";
                }
                else if (!input2_shape.empty() && input2_shape.size() == 1)
                {
                    input2_coords = "[i1]";
                }
                else
                {
                    input2_coords = "[i0][i1]";
                }
                writer << "temp[i0][i1] = input2" << input2_coords << " * " << beta << ";\n";
            }
            writer.block_end();
        }
        writer.block_end();

        writer << "const uint i0 = get_global_id(0);";
        gws.push_back(output_shape.at(0));
        writer << "/*trip count " << output_shape.at(0) << "*/\n";
        writer.block_begin();
        {
            writer << "const uint i1 = get_global_id(1);";
            gws.push_back(output_shape.at(1));
            writer << "/*trip count " << output_shape.at(1) << "*/\n";
            writer.block_begin();
            {
                string acc;
                if (type_name == "float")
                {
                    acc = "0.0f";
                }
                else
                {
                    acc = "0.0";
                }
                writer << type_name << " acc = " << acc << ";\n";
                size_t k_coord = transA ? input0_shape.at(0) : input0_shape.at(1);
                writer << "for (uint k=0; k < " << k_coord << "; ++k)\n";
                writer.block_begin();
                {
                    string input0_coord = transA ? "[k][i0]" : "[i0][k]";
                    string input1_coord = transB ? "[i1][k]" : "[k][i1]";
                    writer << "acc += input0" << input0_coord << " * input1" << input1_coord
                           << ";\n";
                }
                writer.block_end();
                writer << "output[i0][i1] = acc * " << alpha << " + temp[i0][i1];\n";
            }
            writer.block_end();
        }
        writer.block_end();
    }
    writer.block_end();

    const CustomKernelInfo krn_ret(output_name,
                                   output_shape,
                                   output_type,
                                   {input0_name, input1_name, input2_name},
                                   {writer.get_code()},
                                   entry_point_name,
                                   gws);
    return {krn_ret};
}

CustomKernels::krnl_info CustomKernels::build_krnl(const shared_ptr<op::Slice>& op) const
{
    const string& input_name = op->get_input_tensor_name(0);
    const Shape& input_shape = op->get_input_shape(0);
    const string& output_name = op->get_output_tensor_name(0);
    const Shape& output_shape = op->get_output_shape(0);
    const element::Type& output_type = op->get_output_element_type(0);
    const Coordinate& lower_bounds = op->get_lower_bounds();
    const Coordinate& uppper_bounds = op->get_upper_bounds();
    const Strides& strides = op->get_strides();
    const string entry_point_name = "slice_" + output_name;
    CodeWriter writer;
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

    const CustomKernelInfo krn_ret(output_name,
                                   output_shape,
                                   output_type,
                                   {input_name},
                                   {writer.get_code()},
                                   entry_point_name,
                                   gws);
    return {krn_ret};
}

void runtime::intelgpu::do_concat_operation(cldnn::topology& topology,
                                            const vector<string>& input_names,
                                            const vector<Shape>& input_shapes,
                                            const string& output_name,
                                            const Shape& output_shape,
                                            const element::Type& output_type,
                                            size_t concat_axis)
{
    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const string kernel_type_name = get_opencl_type_name(output_type);
    string entry_point_name = "concat_" + output_name;

    size_t bound_below = 0;
    size_t idx = 0;
    vector<string>::const_iterator input_name = input_names.cbegin();
    string aux_output_name;

    // this is quite non optimal because cldnn::custom_gpu_primitive
    // does not provide an ability to run kernels simultaneously with the same output
    // Also, need to make a chain of kernels to put kernel0::output0 as kernel1::input1
    // with output name kernel1::output2
    for (auto const& input_shape : input_shapes)
    {
        string name_suffix = to_string(idx);
        const string entry_point_name_suffix = entry_point_name + "_" + name_suffix;
        CodeWriter writer;
        vector<size_t> gws;

        if (idx == 0)
        {
            gen_func_def(writer,
                         entry_point_name_suffix,
                         {kernel_type_name},
                         {input_shape},
                         kernel_type_name,
                         output_shape);
        }
        else
        {
            gen_func_def(writer,
                         entry_point_name_suffix,
                         {2, kernel_type_name},
                         {input_shape, output_shape},
                         kernel_type_name,
                         output_shape);
        }

        writer.block_begin();
        {
            // Main loops
            gws = generate_loops(writer, output_shape, true);

            writer << kernel_type_name << " input_element;\n";

            size_t bound_upper = input_shape.at(concat_axis);

            // copy corresponding elements of input0 into output
            writer << "if (((" << bound_below << " + 0) <= i" << concat_axis << ") && (i"
                   << concat_axis << " < (" << bound_below << " + " << bound_upper << ")))\n";
            writer.block_begin();
            {
                writer << "input_element = input0";

                if (input_shape.empty())
                {
                    // it means scalar
                    writer << "[0]";
                }
                else
                {
                    size_t var_idx = 0;
                    for (auto const i : input_shape)
                    {
                        if (var_idx == concat_axis)
                        {
                            writer << "[i" << var_idx << " - " << bound_below << "]";
                        }
                        else
                        {
                            writer << "[i" << var_idx << "]";
                        }
                        ++var_idx;
                    }
                }
                writer << ";\n";
            }
            writer.block_end();

            // if not a first kernel, copy input1 into output
            if (idx != 0)
            {
                writer << "else\n";
                writer.block_begin();
                {
                    writer << "input_element = input1" << access_dims(output_shape) << ";\n";
                }
                writer.block_end();
            }
            bound_below += bound_upper;

            writer << "output" << access_dims(output_shape) << " = input_element;\n";

            // Closing brackets for main loops
            generate_loops(writer, output_shape, false);
        }
        writer.block_end();

        vector<cldnn::primitive_id> kernel_input;
        vector<cldnn_arg> kernel_arguments;

        kernel_input.push_back(*input_name);
        if (idx == 0)
        {
            kernel_arguments = get_kernel_args(1, 1);
        }
        else
        {
            kernel_input.push_back(aux_output_name);
            kernel_arguments = get_kernel_args(2, 1);
        }

        // last kernel should produce the output name as overall node required
        if (idx == input_shapes.size() - 1)
        {
            name_suffix = "";
        }

        const cldnn::custom_gpu_primitive op_concat(output_name + name_suffix,
                                                    kernel_input,
                                                    {writer.get_code()},
                                                    entry_point_name_suffix,
                                                    kernel_arguments,
                                                    "",
                                                    layout,
                                                    gws);
        topology.add(op_concat);

        ++input_name;
        ++idx;
        aux_output_name = output_name + name_suffix;
    }
}

CustomKernels::krnl_info CustomKernels::build_krnl(const shared_ptr<op::Select>& op) const
{
    const string& input0_name = op->get_input_tensor_name(0);
    const Shape& input0_shape = op->get_input_shape(0);
    const element::Type& input0_type = op->get_input_element_type(0);
    const string& input1_name = op->get_input_tensor_name(1);
    const Shape& input1_shape = op->get_input_shape(1);
    const element::Type& input1_type = op->get_input_element_type(1);
    const string& input2_name = op->get_input_tensor_name(2);
    const Shape& input2_shape = op->get_input_shape(2);
    const element::Type& input2_type = op->get_input_element_type(2);
    const string& output_name = op->get_output_tensor_name(0);
    const Shape& output_shape = op->get_output_shape(0);
    const element::Type& output_type = op->get_output_element_type(0);
    const string entry_point_name = "select_" + output_name;
    CodeWriter writer;
    vector<size_t> gws;

    gen_func_def(writer,
                 entry_point_name,
                 {get_opencl_type_name(input0_type),
                  get_opencl_type_name(input1_type),
                  get_opencl_type_name(input2_type)},
                 {input0_shape, input1_shape, input2_shape},
                 get_opencl_type_name(output_type),
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

    const CustomKernelInfo krn_ret(output_name,
                                   output_shape,
                                   output_type,
                                   {input0_name, input1_name, input2_name},
                                   {writer.get_code()},
                                   entry_point_name,
                                   gws);
    return {krn_ret};
}

static CustomKernels::krnl_info do_logic_kernel(const shared_ptr<Node>& op, const string& operation)
{
    const string& input0_name = op->get_input_tensor_name(0);
    const Shape& input0_shape = op->get_input_shape(0);
    const element::Type& input0_type = op->get_input_element_type(0);
    const string& input1_name = op->get_input_tensor_name(1);
    const Shape& input1_shape = op->get_input_shape(1);
    const string& output_name = op->get_output_tensor_name(0);
    const Shape& output_shape = op->get_output_shape(0);
    const element::Type& output_type = op->get_output_element_type(0);
    const string entry_point_name = "logic_" + output_name;
    CodeWriter writer;
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

    const CustomKernelInfo op_logical(output_name,
                                      output_shape,
                                      output_type,
                                      {input0_name, input1_name},
                                      {writer.get_code()},
                                      entry_point_name,
                                      gws);
    return {op_logical};
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
                                          const string& operation,
                                          bool function_operation)
{
    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const string entry_point_name = "eltwise_" + output_name;
    CodeWriter writer;
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

        writer << "output" << access_dims(output_shape) << " = ";
        if (function_operation)
        {
            string explicit_conversion;
            // TODO need better workaround for this built_in
            if (operation == "pow")
            {
                explicit_conversion = "convert_double";
            }

            writer << operation << "(" << explicit_conversion << "(input0"
                   << access_dims(input0_shape) << "), " << explicit_conversion << "(input1"
                   << access_dims(input1_shape) << "));";
        }
        else
        {
            writer << "(input0" << access_dims(input0_shape) << " " << operation << " input1"
                   << access_dims(input1_shape) << ");";
        }
        writer << " // " << get_opencl_type_name(input0_type) << " "
               << get_opencl_type_name(output_type) << "\n";

        // Closing brackets for main loops
        generate_loops(writer, output_shape, false);
    }
    writer.block_end();

    const cldnn::custom_gpu_primitive op_eltwise(output_name,
                                                 {input0_name, input1_name},
                                                 {writer.get_code()},
                                                 entry_point_name,
                                                 get_kernel_args(2, 1),
                                                 "",
                                                 layout,
                                                 gws);
    topology.add(op_eltwise);
}

void runtime::intelgpu::do_relu_backprop(cldnn::topology& topology,
                                         const string& input0_name,
                                         const Shape& input0_shape,
                                         const element::Type& input0_type,
                                         const string& input1_name,
                                         const Shape& input1_shape,
                                         const string& output_name,
                                         const Shape& output_shape,
                                         const element::Type& output_type)
{
    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const string entry_point_name = "relubackprop_" + output_name;
    const string input0_type_name = get_opencl_type_name(input0_type);
    const string output_type_name = get_opencl_type_name(output_type);
    const string zero_input0_const = "convert_" + input0_type_name + "(0)";
    const string zero_output_const = "convert_" + output_type_name + "(0)";

    CodeWriter writer;
    vector<size_t> gws;

    gen_func_def(writer,
                 entry_point_name,
                 {2, input0_type_name},
                 {input0_shape, input1_shape},
                 output_type_name,
                 output_shape);

    writer.block_begin();
    {
        // Main loops
        gws = generate_loops(writer, output_shape, true);

        writer << "output" << access_dims(output_shape) << " = (input0" << access_dims(input0_shape)
               << " > " << zero_input0_const << ") ? input1" << access_dims(input1_shape) << " : "
               << zero_output_const << ";\n";

        // Closing brackets for main loops
        generate_loops(writer, output_shape, false);
    }
    writer.block_end();

    const cldnn::custom_gpu_primitive op_reluback(output_name,
                                                  {input0_name, input1_name},
                                                  {writer.get_code()},
                                                  entry_point_name,
                                                  get_kernel_args(2, 1),
                                                  "",
                                                  layout,
                                                  gws);
    topology.add(op_reluback);
}

void runtime::intelgpu::do_reverse_operation(cldnn::topology& topology,
                                             const string& input_name,
                                             const Shape& input_shape,
                                             const element::Type& input_type,
                                             const string& output_name,
                                             const Shape& output_shape,
                                             const element::Type& output_type,
                                             const AxisSet& reversed_axes)
{
    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const string entry_point_name = "reverse_" + output_name;
    CodeWriter writer;
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

void runtime::intelgpu::do_reverse_sequence_operation(cldnn::topology& topology,
                                                      const string& input0_name,
                                                      const Shape& input0_shape,
                                                      const element::Type& input0_type,
                                                      const string& input1_name,
                                                      const Shape& input1_shape,
                                                      const element::Type& input1_type,
                                                      const string& output_name,
                                                      const Shape& output_shape,
                                                      const element::Type& output_type,
                                                      const size_t reversed_axis,
                                                      const size_t batch_axis)
{
    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const string entry_point_name = "reverse_sequence_" + output_name;
    CodeWriter writer;
    vector<size_t> gws;

    gen_func_def(writer,
                 entry_point_name,
                 {get_opencl_type_name(input0_type), get_opencl_type_name(input1_type)},
                 {input0_shape, input1_shape},
                 get_opencl_type_name(output_type),
                 output_shape);

    writer.block_begin();
    {
        writer << "//reversed_axis:" << reversed_axis << "\n";
        writer << "//batch_axis:" << batch_axis << "\n\n";

        gws = generate_loops(writer, output_shape, true);

        writer << get_opencl_type_name(input1_type) << " orig_seq_index = "
               << "input1[i" << batch_axis << "];\n";
        writer << "if (orig_seq_index == 0)\n";
        writer.block_begin();
        {
            writer << "orig_seq_index = 1;\n";
        }
        writer.block_end();

        writer << get_opencl_type_name(input1_type) << " sequence_index;\n";
        writer << "if (i" << reversed_axis << " < orig_seq_index)\n";
        writer.block_begin();
        {
            writer << "sequence_index = orig_seq_index - i" << reversed_axis << " - 1;\n";
        }
        writer.block_end();
        writer << "else\n";
        writer.block_begin();
        {
            writer << "sequence_index = i" << reversed_axis << ";\n";
        }
        writer.block_end();

        writer << "output" << access_dims(output_shape) << " = input0";

        if (output_shape.empty())
        {
            writer << "[0]";
        }
        else
        {
            size_t var_idx = 0;
            for (auto const& i : output_shape)
            {
                if (var_idx == reversed_axis)
                {
                    writer << "[sequence_index]";
                }
                else
                {
                    writer << "[i" << var_idx << "]";
                }
                ++var_idx;
            }
        }
        writer << ";\n";

        generate_loops(writer, output_shape, false);
    }
    writer.block_end();

    const cldnn::custom_gpu_primitive op_reverse_seq(output_name,
                                                     {input0_name, input1_name},
                                                     {writer.get_code()},
                                                     entry_point_name,
                                                     get_kernel_args(2, 1),
                                                     "",
                                                     layout,
                                                     gws);
    topology.add(op_reverse_seq);
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
    CodeWriter writer;
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
    CodeWriter writer;
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

static string emit_convert_bool(const string& input_type)
{
    CodeWriter writer;

    writer << "bool convert_bool(const " << input_type << " input)";
    writer.block_begin();
    {
        writer << "if (input)\n";
        writer.block_begin();
        {
            writer << "return 1;\n";
        }
        writer.block_end();
        writer << "else\n";
        writer.block_begin();
        {
            writer << "return 0;\n";
        }
        writer.block_end();
    }
    writer.block_end();

    return writer.get_code();
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
    CodeWriter writer;
    vector<size_t> gws;

    if (output_type == element::Type_t::boolean)
    {
        writer << emit_convert_bool(input_type_name);
    }

    gen_func_def(
        writer, entry_point_name, {input_type_name}, {input_shape}, output_type_name, output_shape);

    writer.block_begin();
    {
        gws = generate_loops(writer, output_shape, true);

        if (((input_type.get_type_enum() == element::Type_t::f64) ||
             (input_type.get_type_enum() == element::Type_t::f32)) &&
            (output_type.get_type_enum() != element::Type_t::boolean))
        {
            // this is the workaround for OpenCL to be same as with CPU floating point operations
            writer << input_type_name << " input_var = input0" << access_dims(output_shape) << ";\n"
                   << output_type_name << " output_var = 0;\n";

            writer << "if (input_var > " << get_opencl_type_min_max_value(output_type, false);
            if (!output_type.is_real())
            {
                writer << " || isnan(input_var)";
            }
            writer << ")\n";
            writer.block_begin();
            {
                writer << "output_var = " << get_opencl_type_min_max_value(output_type, true)
                       << ";\n";
            }
            writer.block_end();

            writer << "else\n";

            writer.block_begin();
            {
                writer << "output_var = convert_" << output_type_name << "(input_var);\n";
            }
            writer.block_end();

            writer << "output" << access_dims(output_shape) << " = output_var;\n";
        }
        else
        {
            writer << "output" << access_dims(output_shape) << " = convert_" << output_type_name
                   << "(input0" << access_dims(output_shape) << ");\n";
        }

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
    CodeWriter writer;
    vector<size_t> gws;

    gen_func_def(writer,
                 entry_point_name,
                 {2, get_opencl_type_name(output_type)},
                 {input_shape, delta_shape},
                 get_opencl_type_name(output_type),
                 output_shape);

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

void runtime::intelgpu::do_custom_unary_operation(cldnn::topology& topology,
                                                  const string& input_name,
                                                  const Shape& input_shape,
                                                  const element::Type& input_type,
                                                  const string& output_name,
                                                  const Shape& output_shape,
                                                  const element::Type& output_type,
                                                  const string& operation_name)
{
    const string entry_point_name = "op_custom_unary_" + output_name;
    const string intermidiate_type = input_type.size() < 8 ? "float" : "double";
    CodeWriter writer;
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

        // convert to intermediate floating point type
        writer << intermidiate_type << " input_var = convert_" << intermidiate_type << "(input0"
               << access_dims(input_shape) << ");\n";

        // do the operation with the same type
        writer << intermidiate_type << " output_var = " << operation_name
               << "; //Type: " << get_opencl_type_name(input_type) << "\n";

        // convert to destination type
        writer << "output" << access_dims(output_shape) << " = convert_"
               << get_opencl_type_name(output_type) << "(output_var);\n";

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
    CodeWriter writer;
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

void runtime::intelgpu::do_reshape_operation(cldnn::topology& topology,
                                             const string& input_name,
                                             const Shape& input_shape,
                                             const element::Type& input_type,
                                             const string& output_name,
                                             const Shape& output_shape,
                                             const element::Type& output_type,
                                             const AxisVector& reshape_axes)
{
    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const string entry_point_name = "reshape_" + output_name;

    // Workaround on openCL bool datatype. Need to be the same as CPU
    const string& input_type_name =
        (input_type == element::Type_t::boolean) ? "char" : get_opencl_type_name(input_type);
    const string& output_type_name =
        (output_type == element::Type_t::boolean) ? "char" : get_opencl_type_name(output_type);
    const size_t dst_shape_size = shape_size(output_shape);
    CodeWriter writer;

    gen_func_def(writer,
                 entry_point_name,
                 {input_type_name},
                 {input_shape},
                 output_type_name,
                 {dst_shape_size});

    writer.block_begin();
    {
        writer << "// input: " << input_shape << "\n";
        writer << "//output: " << output_shape << "\n";
        writer << "//axes: " << reshape_axes << "\n\n";
        writer << "uint output_it = 0;\n";

        // Main operation loop
        for (auto const i : reshape_axes)
        {
            writer << "for (uint i" << i << " = 0; i" << i << " < " << input_shape.at(i) << "; ++i"
                   << i << ")\n";
            writer.block_begin();
        }

        writer << "output[output_it] = input0" << access_dims(input_shape) << ";\n"
               << "++output_it;\n";

        // Closing brackets for loop
        for (auto const i : reshape_axes)
        {
            writer.block_end();
        }
    }
    writer.block_end();

    const cldnn::custom_gpu_primitive op_reshape(output_name,
                                                 {input_name},
                                                 {writer.get_code()},
                                                 entry_point_name,
                                                 get_kernel_args(1, 1),
                                                 "",
                                                 layout,
                                                 {1});
    topology.add(op_reshape);
}

void runtime::intelgpu::do_quantize_operation(cldnn::topology& topology,
                                              const string& input0_name,
                                              const Shape& input0_shape,
                                              const element::Type& input0_type,
                                              const string& input1_name,
                                              const Shape& input1_shape,
                                              const string& input2_name,
                                              const Shape& input2_shape,
                                              const string& output_name,
                                              const Shape& output_shape,
                                              const element::Type& output_type,
                                              const AxisSet& axis,
                                              const ngraph::op::Quantize::RoundMode mode)
{
    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const string entry_point_name = "quantize_" + output_name;
    const string real_type_str = get_opencl_type_name(input0_type);
    const string quant_type_str = get_opencl_type_name(output_type);
    CodeWriter writer;
    vector<size_t> gws;

    gen_func_def(writer,
                 entry_point_name,
                 {real_type_str, real_type_str, quant_type_str},
                 {input0_shape, input1_shape, input2_shape},
                 quant_type_str,
                 output_shape);

    writer.block_begin();
    {
        writer << "// " << axis << "\n"
               << "// rounding mode: " << (int)mode << "\n";

        // Main loops
        gws = generate_loops(writer, input0_shape, true);

        // apply scale
        writer << real_type_str << " qvalue = input0" << access_dims(input0_shape) << " / input1"
               << access_dims(input1_shape) << ";\n";

        // round
        switch (mode)
        {
        case ngraph::op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_INFINITY:
        {
            writer << real_type_str << " abs_qvalue = fabs(qvalue);\n"
                   << real_type_str << " abs_qvalue_toward_inf = floor(abs_qvalue + 0.5);\n"
                   << "qvalue = (qvalue < 0.0) ? -abs_qvalue_toward_inf : abs_qvalue_toward_inf;\n";
        }
        break;
        case ngraph::op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_ZERO:
        {
            writer
                << real_type_str << " abs_qvalue = fabs(qvalue);\n"
                << real_type_str << " abs_qvalue_toward_zero = ceil(abs_qvalue - 0.5);\n"
                << "qvalue = (qvalue < 0.0) ? -abs_qvalue_toward_zero : abs_qvalue_toward_zero;\n";
        }
        break;
        case ngraph::op::Quantize::RoundMode::ROUND_NEAREST_UPWARD:
        {
            writer << "qvalue = floor(qvalue + 0.5);\n";
        }
        break;
        case ngraph::op::Quantize::RoundMode::ROUND_NEAREST_DOWNWARD:
        {
            writer << "qvalue = ceil(qvalue - 0.5);\n";
        }
        break;
        case ngraph::op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_EVEN:
        {
            writer << real_type_str << " up_qvalue = floor(qvalue + 0.5);\n"
                   << real_type_str << " dn_qvalue = ceil(qvalue - 0.5);\n"
                   << real_type_str << " rem = fmod(up_qvalue, convert_" << real_type_str
                   << "(2.0));\n"
                   << "qvalue = (rem == 0.0) ? up_qvalue : dn_qvalue;\n";
        }
        break;
        case ngraph::op::Quantize::RoundMode::ROUND_TOWARD_INFINITY:
        {
            writer << real_type_str << " abs_qvalue = fabs(qvalue);\n"
                   << real_type_str << " abs_qvalue_toward_inf = ceil(abs_qvalue);\n"
                   << "qvalue = (qvalue < 0.0) ? -abs_qvalue_toward_inf : abs_qvalue_toward_inf;\n";
        }
        break;
        case ngraph::op::Quantize::RoundMode::ROUND_TOWARD_ZERO:
        {
            writer
                << real_type_str << " abs_qvalue = fabs(qvalue);\n"
                << real_type_str << " abs_qvalue_toward_zero = floor(abs_qvalue);\n"
                << "qvalue = (qvalue < 0.0) ? -abs_qvalue_toward_zero : abs_qvalue_toward_zero;\n";
        }
        break;

        case ngraph::op::Quantize::RoundMode::ROUND_UP: { writer << "qvalue = ceil(qvalue);\n";
        }
        break;
        case ngraph::op::Quantize::RoundMode::ROUND_DOWN: { writer << "qvalue = floor(qvalue);\n";
        }
        break;
        default:
        {
            throw ngraph_error("Unsupported rounding mode '" + to_string((int)mode) +
                               "' in runtime::intelgpu::do_quantize_operation()");
        }
        }

        // apply offset
        writer << "qvalue += input2" << access_dims(input2_shape) << ";\n";

        // cast to output
        writer << "output" << access_dims(output_shape) << " = convert_" << quant_type_str
               << "(qvalue);\n";

        // Closing brackets for main loops
        generate_loops(writer, input0_shape, false);
    }
    writer.block_end();

    const cldnn::custom_gpu_primitive op_quantize(output_name,
                                                  {input0_name, input1_name, input2_name},
                                                  {writer.get_code()},
                                                  entry_point_name,
                                                  get_kernel_args(3, 1),
                                                  "",
                                                  layout,
                                                  gws);
    topology.add(op_quantize);
}

void runtime::intelgpu::do_dequantize_operation(cldnn::topology& topology,
                                                const std::string& input0_name,
                                                const Shape& input0_shape,
                                                const element::Type& input0_type,
                                                const std::string& input1_name,
                                                const Shape& input1_shape,
                                                const element::Type& input1_type,
                                                const std::string& input2_name,
                                                const Shape& input2_shape,
                                                const element::Type& input2_type,
                                                const string& output_name,
                                                const Shape& output_shape,
                                                const element::Type& output_type,
                                                const AxisSet& axis)
{
    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const string entry_point_name = "dequantize_" + output_name;
    CodeWriter writer;
    vector<size_t> gws;

    gen_func_def(writer,
                 entry_point_name,
                 {get_opencl_type_name(input0_type),
                  get_opencl_type_name(input1_type),
                  get_opencl_type_name(input2_type)},
                 {input0_shape, input1_shape, input2_shape},
                 get_opencl_type_name(output_type),
                 output_shape);

    writer.block_begin();
    {
        writer << "// " << axis << "\n";

        // Main loops
        gws = generate_loops(writer, output_shape, true);

        writer << "output" << access_dims(output_shape) << " = ";
        writer << "(input0" << access_dims(input0_shape) << " - input2" << access_dims(input2_shape)
               << ") * input1" << access_dims(input1_shape) << ";\n";

        // Closing brackets for main loops
        generate_loops(writer, output_shape, false);
    }
    writer.block_end();

    const cldnn::custom_gpu_primitive op_dequantize(output_name,
                                                    {input0_name, input1_name, input2_name},
                                                    {writer.get_code()},
                                                    entry_point_name,
                                                    get_kernel_args(3, 1),
                                                    "",
                                                    layout,
                                                    gws);
    topology.add(op_dequantize);
}

void runtime::intelgpu::do_topk_operation(cldnn::topology& topology,
                                          const std::string& input_name,
                                          const Shape& input_shape,
                                          const element::Type& input_type,
                                          const std::string& output_name,
                                          const Shape& output_shape,
                                          const element::Type& output_type,
                                          const element::Type& index_elem_type,
                                          const size_t top_k_axis,
                                          const size_t k,
                                          const bool compute_max,
                                          const bool find_indices)
{
    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const string entry_point_name = "topk_" + output_name;
    CodeWriter writer;
    const string operation_sign = compute_max ? " > " : " < ";
    const string prev_operation_sign = !compute_max ? ">" : "<";
    const size_t shape_size = input_shape.size();

    gen_func_def(writer,
                 entry_point_name,
                 {get_opencl_type_name(input_type)},
                 {input_shape},
                 get_opencl_type_name(output_type),
                 output_shape);

    writer.block_begin();
    {
        writer << get_opencl_type_name(input_type)
               << " prev_min_max = " << get_opencl_type_min_max_value(input_type, !compute_max)
               << ";\n";
        writer << get_opencl_type_name(index_elem_type) << " prev_index = -2;\n";
        writer << get_opencl_type_name(input_type)
               << " current_min_max = " << get_opencl_type_min_max_value(input_type, compute_max)
               << ";\n";
        writer << get_opencl_type_name(index_elem_type) << " current_index = -1;\n";

        size_t current_output = 0;
        for (auto const& i : output_shape)
        {
            if (current_output != top_k_axis)
            {
                writer << "for (uint i" << current_output << " = 0; i" << current_output << " < "
                       << i << "; ++i" << current_output << ")\n";
                writer.block_begin();
            }
            ++current_output;
        }

        writer << "prev_min_max = " << get_opencl_type_min_max_value(input_type, !compute_max)
               << ";\n";
        writer << "prev_index = -2;\n";

        writer << "for (uint i = 0; i < " << output_shape.at(top_k_axis) << "; ++i)\n";
        writer.block_begin();

        writer << "current_min_max = " << get_opencl_type_min_max_value(input_type, compute_max)
               << ";\n";
        writer << "current_index = -1;\n";

        writer << "for (uint j = 0; j < " << input_shape.at(top_k_axis) << "; ++j)\n";
        writer.block_begin();

        size_t current = 0;
        string buffer;
        for (uint j = 0; j < shape_size; ++j)
        {
            if (j == top_k_axis)
            {
                buffer += "[j]";
            }
            else
            {
                buffer += "[i" + to_string(current) + "]";
            }
            ++current;
        }

        writer << "if (input0" << buffer << operation_sign << "current_min_max)\n";
        writer.block_begin();
        {
            writer << "if (input0" << buffer << " " << prev_operation_sign
                   << " prev_min_max || (input0" << buffer
                   << " == prev_min_max && j > prev_index))\n";
            writer.block_begin();
            {
                writer << "current_min_max = input0" << buffer << ";\n";
                writer << "current_index = j;\n";
            }
            writer.block_end();
        }
        writer.block_end();

        writer.block_end();

        current = 0;
        string outbuffer;
        for (uint j = 0; j < shape_size; ++j)
        {
            if (j == top_k_axis)
            {
                outbuffer += "[i]";
            }
            else
            {
                outbuffer += "[i" + to_string(current) + "]";
            }
            ++current;
        }

        if (find_indices == true)
        {
            writer << "output" << outbuffer << " = current_index;\n";
        }
        else
        {
            writer << "output" << outbuffer << " = current_min_max;\n";
        }
        writer << "prev_min_max = current_min_max;\n";
        writer << "prev_index = current_index;\n";

        writer.block_end();
        current_output = 0;
        for (auto const& i : output_shape)
        {
            if (current_output != top_k_axis)
            {
                writer.block_end();
            }
            ++current_output;
        }
    }
    writer.block_end();

    const cldnn::custom_gpu_primitive op_topk(output_name,
                                              {input_name},
                                              {writer.get_code()},
                                              entry_point_name,
                                              get_kernel_args(1, 1),
                                              "",
                                              layout,
                                              {1});
    topology.add(op_topk);
}

size_t runtime::intelgpu::get_max_memory_rss()
{
    size_t result = 0;
    struct rusage usage;

    if (getrusage(RUSAGE_SELF, &usage) == 0)
    {
        result = usage.ru_maxrss; // the value is in kilobytes

        // aligne result to return bytes
        result *= 1000;
    }

    return result;
}

CustomKernels::krnl_info CustomKernels::build_krnl(const shared_ptr<op::And>& op) const
{
    return do_logic_kernel(op, " && ");
}

CustomKernels::krnl_info CustomKernels::build_krnl(const shared_ptr<op::Equal>& op) const
{
    return do_logic_kernel(op, " == ");
}

CustomKernels::krnl_info CustomKernels::build_krnl(const shared_ptr<op::Greater>& op) const
{
    return do_logic_kernel(op, " > ");
}

CustomKernels::krnl_info CustomKernels::build_krnl(const shared_ptr<op::GreaterEq>& op) const
{
    return do_logic_kernel(op, " >= ");
}

CustomKernels::krnl_info CustomKernels::build_krnl(const shared_ptr<op::Less>& op) const
{
    return do_logic_kernel(op, " < ");
}

CustomKernels::krnl_info CustomKernels::build_krnl(const shared_ptr<op::LessEq>& op) const
{
    return do_logic_kernel(op, " <= ");
}

CustomKernels::krnl_info CustomKernels::build_krnl(const shared_ptr<op::NotEqual>& op) const
{
    return do_logic_kernel(op, " != ");
}

CustomKernels::krnl_info CustomKernels::build_krnl(const shared_ptr<op::Or>& op) const
{
    return do_logic_kernel(op, " || ");
}
