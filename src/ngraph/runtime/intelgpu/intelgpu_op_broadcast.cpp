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

#include <CPP/concatenation.hpp>
#include <CPP/reshape.hpp>

#include "ngraph/runtime/intelgpu/intelgpu_layout.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_broadcast.hpp"

#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

static const string reshape_suf("_reshape");

static Shape propagate_backward(const Shape& input)
{
    Shape result({0, 0, 0, 0});
    size_t idx = result.size() - 1;

    for (auto i = input.crbegin(); i != input.crend(); ++i, --idx)
    {
        result.at(idx) = *i;
    }

    return result;
}

static Shape propagate_forward(const Shape& input)
{
    Shape result({0, 0, 0, 0});
    size_t idx = 0;

    for (auto i = input.cbegin(); i != input.cend(); ++i, ++idx)
    {
        result.at(idx) = *i;
    }

    return result;
}

static Shape apply_axis(const Shape& input, const AxisSet& axis)
{
    Shape result = input;

    for (auto const& i : axis)
    {
        result.at(i) = 0;
    }

    return result;
}

// This function broadcast input data to all other dimensions of the output
// it operates in two mode only (controlled by is_forward flag):
// [forward]: propagate data from left to right in Shape array term
//            in[2], out[2,3,4,5], axis[1,2,3]
// [backward]: propagate data from right to left in Shape array term
//            in[5], out[2,3,4,5], axis[0,1,2]
// Input and output shapes can be up to 4 dimensions
// Other variants, like: in[4] out[2,3,4,5] axis[0,1,3], unsupported yet
static void do_propagation(cldnn::topology& topology,
                           const string& input_name,
                           const Shape& input_shape,
                           const string& output_name,
                           const Shape& output_shape,
                           const AxisSet& axis,
                           bool is_forward)
{
    //default value used in "forward" mode
    cldnn::concatenation::concatenation_axis direction =
        runtime::intelgpu::IntelGPULayout::get_cldnn_axis(3);

    string input_name_it = input_name;
    string output_name_it = output_name;
    Shape input_shape_it = input_shape;
    for (auto axis_id = axis.crbegin(); axis_id != axis.crend();)
    {
        const size_t input_count = output_shape.at(*axis_id);

        if (is_forward)
        {
            input_shape_it.push_back(1);
            const cldnn::tensor my_tensor =
                runtime::intelgpu::IntelGPULayout::create_cldnn_tensor(input_shape_it);

            const cldnn::reshape op_reshape(input_name_it + reshape_suf, input_name_it, my_tensor);
            topology.add(op_reshape);

            input_shape_it.back() = input_count;
            input_name_it += reshape_suf;
        }
        else
        {
            direction = runtime::intelgpu::IntelGPULayout::get_cldnn_axis(*axis_id);
        }

        const vector<cldnn::primitive_id> input_names(input_count, input_name_it);

        ++axis_id;
        if (axis_id == axis.crend())
        {
            output_name_it = output_name;
        }
        else
        {
            output_name_it += ":_";
            input_name_it = output_name_it;
        }

        const cldnn::concatenation op_concat(output_name_it, input_names, direction);
        topology.add(op_concat);
    }
}

// Assume input is scalar. All output data will be populated by the scalar
// The function extremely non optimal from performance perspective
static void do_scalar_propagation(cldnn::topology& topology,
                                  const string& input_name,
                                  const string& output_name,
                                  const Shape& output_shape)
{
    const size_t input_count = shape_size<const Shape>(output_shape);
    const vector<cldnn::primitive_id> input_names(input_count, input_name);

    const cldnn::concatenation op_concat(output_name, input_names, cldnn::concatenation::along_x);
    topology.add(op_concat);
}

void runtime::intelgpu::do_broadcast_operation(cldnn::topology& topology,
                                               const string& input_name,
                                               const Shape& input_shape,
                                               const string& output_name,
                                               const Shape& output_shape,
                                               const AxisSet& axis)
{
    if (input_shape.size() > 4 || output_shape.size() > 4)
    {
        throw invalid_argument("IntelGPU::Broadcast supports 4D shapes maximum.");
    }

    if (input_shape.empty())
    {
        do_scalar_propagation(topology, input_name, output_name, output_shape);

        return;
    }

    const Shape output_shape_axis = apply_axis(output_shape, axis);
    const Shape input_shape_forward = propagate_forward(input_shape);
    const Shape output_shape_forward = propagate_forward(output_shape_axis);
    const Shape input_shape_backward = propagate_backward(input_shape);
    const Shape output_shape_backward = propagate_backward(output_shape_axis);

    if (input_shape_forward == output_shape_forward)
    {
        do_propagation(topology, input_name, input_shape, output_name, output_shape, axis, true);
    }
    else if (input_shape_backward == output_shape_backward)
    {
        do_propagation(topology, input_name, input_shape, output_name, output_shape, axis, false);
    }
    else
    {
        ostringstream os;
        os << "IntelGP::Broadcast unsupported mode. input" << vector_to_string(input_shape)
           << " output" << vector_to_string(output_shape) << " axis" << vector_to_string(axis);
        throw invalid_argument(os.str());
    }
}
