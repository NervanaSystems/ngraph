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

#include <functional>
#include <memory>
#include <utility>

#include "ngraph/axis_vector.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

op::Dot::Dot(const shared_ptr<Node>& arg0, const shared_ptr<Node>& arg1)
    : Dot(arg0, arg1, 0, false)
{
}

op::Dot::Dot(const shared_ptr<Node>& arg0,
             const shared_ptr<Node>& arg1,
             size_t reduction_axes_count,
             bool has_reduction_axes_count)
    : Op("Dot", check_single_output_args({arg0, arg1}))
    , m_reduction_axes_count(reduction_axes_count)
    , m_has_reduction_axes_count(has_reduction_axes_count)
{
    constructor_validate_and_infer_types();
}

void op::Dot::validate_and_infer_types()
{
    auto& input_0 = get_inputs().at(0);
    auto& input_1 = get_inputs().at(1);

    if (!m_has_reduction_axes_count)
    {
        m_reduction_axes_count =
            (input_0.get_shape().size() == 0 || input_1.get_shape().size() == 0) ? 0 : 1;
    }

    NODE_VALIDATION_ASSERT(this, input_0.get_element_type() == input_1.get_element_type())
        << "Arguments do not have the same element type (arg0 element type: "
        << input_0.get_element_type() << ", arg1 element type: " << input_1.get_element_type()
        << ").";

    Shape input_0_shape = input_0.get_shape();
    Shape input_1_shape = input_1.get_shape();

    NODE_VALIDATION_ASSERT(this,
                           m_reduction_axes_count <= input_0_shape.size() &&
                               m_reduction_axes_count <= input_1_shape.size())
        << "Reduction axes count (" << m_reduction_axes_count
        << ") is too large (arg0 shape: " << input_0_shape << ", arg1 shape: " << input_1_shape
        << ").";

    for (size_t i = 0; i < m_reduction_axes_count; i++)
    {
        size_t axis_index_arg0 = input_0_shape.size() - m_reduction_axes_count + i;
        size_t axis_index_arg1 = i;

        NODE_VALIDATION_ASSERT(this,
                               input_0_shape[axis_index_arg0] == input_1_shape[axis_index_arg1])
            << "Paired axes (axis " << axis_index_arg0 << " from arg0, axis " << axis_index_arg1
            << " from arg1) "
            << "do not have same length (arg0 shape: " << input_0_shape
            << ", arg1 shape: " << input_1_shape << ", "
            << "reduction axes count: " << m_reduction_axes_count << ").";
    }

    Shape result_shape(input_0_shape.size() + input_1_shape.size() - 2 * m_reduction_axes_count);

    copy(input_0_shape.begin(), input_0_shape.end() - m_reduction_axes_count, result_shape.begin());
    copy(input_1_shape.begin() + m_reduction_axes_count,
         input_1_shape.end(),
         result_shape.begin() + (input_0_shape.size() - m_reduction_axes_count));

    set_output_type(0, input_0.get_element_type(), result_shape);
}

shared_ptr<op::Reshape> make_reshape_axes_to_front(const shared_ptr<Node>& n,
                                                   const Shape& front_shape,
                                                   const Shape& back_shape)
{
    AxisVector input_order;
    Shape output_shape;

    for (size_t i = 0; i < back_shape.size(); i++)
    {
        input_order.push_back(front_shape.size() + i);
        output_shape.push_back(back_shape[i]);
    }

    for (size_t i = 0; i < front_shape.size(); i++)
    {
        input_order.push_back(i);
        output_shape.push_back(front_shape[i]);
    }

    return make_shared<op::Reshape>(n, input_order, output_shape);
}

void op::Dot::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = get_inputs().at(0).get_output().get_node();
    auto y = get_inputs().at(1).get_output().get_node();

    auto x_shape = x->get_shape();         // shape IJ
    auto y_shape = y->get_shape();         // shape JK
    auto delta_shape = delta->get_shape(); // shape IK

    Shape I_shape;
    Shape J_shape;
    Shape K_shape;
    I_shape.insert(I_shape.begin(), x_shape.begin(), x_shape.end() - m_reduction_axes_count);
    J_shape.insert(J_shape.begin(), y_shape.begin(), y_shape.begin() + m_reduction_axes_count);
    K_shape.insert(K_shape.begin(), y_shape.begin() + J_shape.size(), y_shape.end());

    auto y_reshaped = make_reshape_axes_to_front(y, J_shape, K_shape);               // KI
    auto delta_dot_y_reshaped = make_shared<Dot>(delta, y_reshaped, K_shape.size()); // JI
    adjoints.add_delta(x, delta_dot_y_reshaped);

    auto x_reshaped = make_reshape_axes_to_front(x, I_shape, J_shape);               // JI
    auto x_reshaped_dot_delta = make_shared<Dot>(x_reshaped, delta, I_shape.size()); // JK
    adjoints.add_delta(y, x_reshaped_dot_delta);
}
