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
#include "ngraph/runtime/intelgpu/intelgpu_op_convolution.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_custom_kernels.hpp"

using namespace std;
using namespace ngraph;

// this is duplication of the runtime::intelgpu::access_dims
// needs to be merged but not at the same time as this new code
static string array_dim(const Shape& dimentions, const string& var = "i", bool is_reversed = false)
{
    size_t var_idx = 0;
    string buffer;

    for (auto const& i : dimentions)
    {
        if (is_reversed)
        {
            buffer += "[" + to_string(i) + " - " + var + to_string(var_idx) + " - 1]";
        }
        else
        {
            buffer += "[" + var + to_string(var_idx) + "]";
        }
        ++var_idx;
    }

    if (buffer.empty())
    { // it means scalar
        buffer = "[0]";
    }

    return buffer;
}

// Padding, Strides and dilation are quite nice explained
// with animations here https://github.com/vdumoulin/conv_arithmetic
//
// batch axes for both input data and output data are 0
// input channel axes for both input data and filters are 1
// output channel axes for filters is 0
// output channel axis for output data is 1
//
// Example (Convolution):
//           data[ 2, 1, 3, 5, 8 ]
//         filter[ 2, 1, 2, 2, 3 ]
//         output[ 2, 2, 2, 4, 6 ]
// it is like
//           data[          batch,   data_channel, 3, 5, 8 ]
//         filter[ output_channel,   data_channel, 2, 2, 3 ]
//         output[          batch, output_channel, 2, 4, 6 ]
//
// Example (ConvolutionBackpropFilters):
//           data[ 2, 1, 3, 5 ]
//         filter[ 2, 2, 2, 4 ]
//         output[ 2, 1, 2, 2 ]
// it is like
//           data[   data_channel,          batch, 3, 5 ]
//         filter[   data_channel, output_channel, 2, 4 ]
//         output[ output_channel,          batch, 2, 2 ]
//
// Example (ConvolutionBackpropData):
//           data[ 2, 2, 2, 4 ]
//         filter[ 2, 1, 2, 2 ]
//         output[ 2, 1, 3, 5 ]
//      pad_below[ 1, 1 ]
//      pad_above[ 1, 1 ]
// it is like
//           data[         batch,   data_channel, 2, 4 ]
//         filter[  data_channel, output_channel, 2, 2 ]
//         output[         batch, output_channel, 3, 5 ]
void runtime::intelgpu::do_convolution_operation(cldnn::topology& topology,
                                                 const string& input_name,
                                                 const Shape& input_shape,
                                                 const string& filter_name,
                                                 const Shape& filter_shape,
                                                 const string& output_name,
                                                 const Shape& output_shape,
                                                 const element::Type& output_type,
                                                 const CoordinateDiff& pad_below,
                                                 const Strides& win_stride,
                                                 const Strides& win_dilation,
                                                 const Strides& data_dilation,
                                                 size_t batch_axis_data,
                                                 size_t input_channel_axis_data,
                                                 size_t output_channel_axis_result,
                                                 const string& input_order,
                                                 const string& filter_order,
                                                 const string& output_order,
                                                 bool reverse_filter)
{
    const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(output_type, output_shape);
    const string entry_point_name = "convolution_" + output_name;
    const Shape input_data(input_shape.cbegin() + 2, input_shape.cend());
    const Shape filter_data(filter_shape.cbegin() + 2, filter_shape.cend());
    const Shape output_data(output_shape.cbegin() + 2, output_shape.cend());
    codegen::CodeWriter writer;
    vector<size_t> gws;

    writer << "__kernel void " << entry_point_name << "(const __global float input"
           << array_dims(input_shape) << ", const __global float filter" << array_dims(filter_shape)
           << ", __global float output" << array_dims(output_shape) << ")\n";

    writer.block_begin();
    { // Main function body

        writer << "const unsigned batch = get_global_id(0);\n";
        gws.push_back(output_shape.at(batch_axis_data));
        writer << "// for (uint batch = 0; batch < " << output_shape.at(batch_axis_data)
               << "; ++batch)\n";
        writer.block_begin();
        {
            writer << "const unsigned output_channel = get_global_id(1);\n";
            gws.push_back(output_shape.at(output_channel_axis_result));
            writer << "// for (uint output_channel = 0; output_channel < "
                   << output_shape.at(output_channel_axis_result) << "; ++output_channel)\n";
            writer.block_begin();
            {
                // The first loop over output dimensions
                writer << "const unsigned i0 = get_global_id(2);\n";
                gws.push_back(output_data.at(0));
                writer << "// for (uint i0 = 0; i0 < " << output_data.at(0) << "; ++i0)\n";
                writer.block_begin();
                {
                    // Loops over other output dimensions
                    size_t var_idx = 1;
                    for (auto i = output_data.begin() + 1; i != output_data.end(); ++i)
                    {
                        writer << "for (uint i" << var_idx << " = 0; i" << var_idx << " < " << *i
                               << "; ++i" << var_idx << ")\n";
                        writer.block_begin();

                        ++var_idx;
                    }

                    writer << "float result = 0.0f;\n\n"
                           << "// Loop over input_channel\n"
                           << "for (uint input_channel = 0; input_channel < "
                           << input_shape.at(input_channel_axis_data) << "; ++input_channel)\n";
                    writer.block_begin();
                    {
                        // Loop over filter
                        // Since first two dimensions are special, let start from third dimension
                        writer << "// Over filter iterations\n";

                        var_idx = 0;
                        for (auto const& i : filter_data)
                        {
                            writer << "for (uint f" << var_idx << " = 0; f" << var_idx << " < " << i
                                   << "; ++f" << var_idx << ")\n";
                            writer.block_begin();

                            writer << "uint input_idx" << var_idx << " = (i" << var_idx << " * "
                                   << win_stride.at(var_idx) << " /*win_stride*/"
                                   << ") + (f" << var_idx << " * " << win_dilation.at(var_idx)
                                   << " /*win_dilation*/)"
                                   << " - " << pad_below.at(var_idx) << " /*pad_below*/;\n";

                            writer << "uint input_idx_data_dilation" << var_idx << " = input_idx"
                                   << var_idx << " / " << data_dilation.at(var_idx)
                                   << " /*data_dilation*/;\n";

                            ++var_idx;
                        }

                        // Generate dilation conditionals
                        writer << "if (";
                        var_idx = 0;
                        for (auto const& i : output_data)
                        {
                            if (var_idx)
                            {
                                writer << " && ";
                            }

                            writer << "(((input_idx" << var_idx << ") % "
                                   << data_dilation.at(var_idx) << " /*data_dilation*/) == 0)";

                            ++var_idx;
                        }
                        writer << ")  /*data_dilation. If we are in a dilation gap"
                                  ", we have no source coordinate.*/\n";
                        writer.block_begin();
                        {
                            // Generate other conditionals
                            writer << "// Since we use unsigned indexes we don't need "
                                   << "(input_idx_data_dilationX >= 0) extra check\n"
                                   << "if (";
                            var_idx = 0;
                            for (auto const& i : input_data)
                            {
                                if (var_idx)
                                {
                                    writer << " && ";
                                }

                                writer << "(input_idx_data_dilation" << var_idx << " < " << i
                                       << ")";

                                ++var_idx;
                            }
                            writer << ")\n";
                            writer.block_begin();
                            {
                                writer << "float input_elem = " << input_order
                                       << array_dim(input_data, "input_idx_data_dilation") << ";\n";

                                // Output element calculation
                                writer << "result += input_elem * " << filter_order
                                       << array_dim(filter_data, "f", reverse_filter) << ";\n";
                            }
                            writer.block_end();
                            // End of other conditional generation
                        }
                        writer.block_end();
                        // End of dilation conditional generation

                        // Closing brackets for filter loop
                        for (auto const& i : filter_data)
                        {
                            writer.block_end();
                        }
                    }
                    writer.block_end();
                    writer << "// End input_channel loop\n";

                    writer << output_order << access_dims(output_data) << " = result;\n";

                    // Closing brackets for other output dimensions
                    for (auto i = output_data.begin() + 1; i != output_data.end(); ++i)
                    {
                        writer.block_end();
                    }
                } // Closing brackets for the first loop over output dimensions
                writer.block_end();
            } // End of loop over output_channel
            writer.block_end();
        } // End of loop over batch
        writer.block_end();

    } // Main function body
    writer.block_end();

    const cldnn::custom_gpu_primitive op_convolution(output_name,
                                                     {input_name, filter_name},
                                                     {writer.get_code()},
                                                     entry_point_name,
                                                     get_kernel_args(2, 1),
                                                     "",
                                                     layout,
                                                     gws);
    topology.add(op_convolution);
}
