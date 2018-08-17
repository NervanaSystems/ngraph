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

//
// Helper function to compute the number of dot axes according to default behavior when
// they are not specified.
//
size_t default_reduction_axes_count(const shared_ptr<Node>& arg0, const shared_ptr<Node>& arg1)
{
    if (arg0->get_shape().size() == 0 || arg1->get_shape().size() == 0)
    {
        return 0;
    }
    else
    {
        return 1;
    }
}

op::Dot::Dot(const shared_ptr<Node>& arg0, const shared_ptr<Node>& arg1)
    : Dot(arg0, arg1, default_reduction_axes_count(arg0, arg1))
{
}

op::Dot::Dot(const shared_ptr<Node>& arg0,
             const shared_ptr<Node>& arg1,
             size_t reduction_axes_count)
    : RequiresTensorViewArgs("Dot", {arg0, arg1})
    , m_reduction_axes_count(reduction_axes_count)
{
    auto& input_0 = get_inputs().at(0);
    auto& input_1 = get_inputs().at(1);

    if (input_0.get_element_type() != input_1.get_element_type())
    {
        throw ngraph_error("Arguments to dot must have the same element type");
    }

    Shape input_0_shape = input_0.get_shape();
    Shape input_1_shape = input_1.get_shape();

    if (reduction_axes_count > input_0_shape.size())
    {
        throw ngraph_error("Dot has too many axes for arg0");
    }

    if (reduction_axes_count > input_1_shape.size())
    {
        throw ngraph_error("Dot has too many axes for arg1");
    }

    for (size_t i = 0; i < reduction_axes_count; i++)
    {
        if (input_0_shape[input_0_shape.size() - reduction_axes_count + i] != input_1_shape[i])
        {
            throw ngraph_error("Dot axes do not have same length");
        }
    }

    Shape result_shape(input_0_shape.size() + input_1_shape.size() - 2 * reduction_axes_count);

    copy(input_0_shape.begin(), input_0_shape.end() - reduction_axes_count, result_shape.begin());
    copy(input_1_shape.begin() + reduction_axes_count,
         input_1_shape.end(),
         result_shape.begin() + (input_0_shape.size() - reduction_axes_count));

    auto result_type = make_shared<TensorViewType>(input_0.get_element_type(), result_shape);
    set_value_type_checked(result_type);
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
