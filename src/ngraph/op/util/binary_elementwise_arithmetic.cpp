//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/log.hpp"

using namespace std;
using namespace ngraph;

op::util::BinaryElementwiseArithmetic::BinaryElementwiseArithmetic(const AutoBroadcastSpec& autob)
    : m_autob(autob)
{
}

op::util::BinaryElementwiseArithmetic::BinaryElementwiseArithmetic(const Output<Node>& arg0,
                                                                   const Output<Node>& arg1,
                                                                   const AutoBroadcastSpec& autob)
    : Op({arg0, arg1})
    , m_autob(autob)
{
}

void op::util::BinaryElementwiseArithmetic::validate_and_infer_types()
{
    validate_and_infer_elementwise_arithmetic(m_autob);
}

bool op::util::BinaryElementwiseArithmetic::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("auto_broadcast", m_autob);
    return true;
}

Shape op::util::BinaryElementwiseArithmetic::compute_output_shape(const Shape& arg0_shape,
                                                                  const Shape& arg1_shape) const
{
    Shape output_shape;
    switch (m_autob.m_type)
    {
    case op::AutoBroadcastType::NONE:
        output_shape = arg0_shape;
        break;
    case op::AutoBroadcastType::NUMPY:
        // We'll be using CoordinateTransform to handle the broadcasting. The general
        // procedure is as follows:
        //
        // (1) Left pad the shorter of the two shapes with ones.
        // (2) Squeeze (remove ones from) both shapes, and record the squeezed axis
        //     indices.
        // (3) Using CoordinateTransform, broadcast both args to the final output
        //     shape. The "broadcasted axes" will be those that were squeezed in step
        //     2.
        //
        // Example:
        //
        //    Input shape->Padded shape->Squeezed Shape/Squeezed Axes
        //    -----------  ------------  ----------------------------
        // a: [ 3, 2, 1]   [ 3, 2, 1]    [ 3, 2   ]     {2}
        // b: [    1, 6]   [ 1, 1, 6]    [       6]     {0,1}
        //                   |  |  |
        //                   v  v  v
        //                 Output shape
        //                 ------------
        //                 [ 3, 2, 6]
        {
            Shape arg0_padded_shape = arg0_shape;
            Shape arg1_padded_shape = arg1_shape;

            while (arg0_padded_shape.size() < arg1_padded_shape.size())
            {
                arg0_padded_shape.insert(arg0_padded_shape.begin(), 1);
            }

            while (arg1_padded_shape.size() < arg0_padded_shape.size())
            {
                arg1_padded_shape.insert(arg1_padded_shape.begin(), 1);
            }

            for (size_t i = 0; i < arg0_padded_shape.size(); i++)
            {
                output_shape.push_back(arg0_padded_shape[i] == 1 ? arg1_padded_shape[i]
                                                                 : arg0_padded_shape[i]);
            }
        }
        break;
    case op::AutoBroadcastType::PDPD:
        // We'll be using CoordinateTransform to handle the broadcasting. No need to
        // process arg0 and output shape will be the same as arg0. We need to process
        // arg1 and the general procedure is as follows:
        //
        // (1) Trim trailing ones from arg1 shape.
        // (2) Left and right pad arg1 to match arg0 shape. Axis is the index start
        //     to align between arg0 and arg1.
        // (3) Squeeze (remove ones from) arg1 shape, and record the squeezed axis
        //     indices.
        // (3) Using CoordinateTransform, broadcast arg1 to the final output
        //     shape. The "broadcasted axes" will be those that were squeezed in step
        //     23.
        //
        // Example:
        //
        //    Input shape->   Padded shape->   Squeezed Shape/Squeezed Axes
        //    -----------  ------------  ----------------------------
        // a: [ 3, 4, 5, 6]   [ 3, 4, 5, 6]    [ 3, 4, 5, 6]
        // b: [    4, 5,  ]   [ 1, 4, 5, 1]    [    4, 5   ]     {0,3}
        //                      |  |  |
        //                      v  v  v
        //                     Output shape
        //                     ------------
        //                    [ 3, 4, 5, 6]
        {
            int64_t axis = m_autob.m_axis;
            if (axis == -1)
            {
                axis = arg0_shape.size() - arg1_shape.size();
            }

            Shape arg1_padded_shape = arg1_shape;
            // Trim trailing ones
            while (arg1_padded_shape.size() > 0 && arg1_padded_shape.back() == 1)
            {
                arg1_padded_shape.pop_back();
            }

            for (int64_t i = 0; i < axis; ++i)
            {
                arg1_padded_shape.insert(arg1_padded_shape.begin(), 1);
            }

            while (arg1_padded_shape.size() < arg0_shape.size())
            {
                arg1_padded_shape.insert(arg1_padded_shape.end(), 1);
            }

            for (size_t i = 0; i < arg0_shape.size(); i++)
            {
                if (arg1_padded_shape[i] == 1)
                {
                    output_shape.push_back(arg0_shape[i]);
                }
                else
                {
                    output_shape.push_back(arg1_padded_shape[i]);
                }
            }
        }
    }
    return output_shape;
}
