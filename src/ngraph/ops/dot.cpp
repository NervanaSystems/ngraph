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

#include <functional>
#include <memory>
#include <utility>

#include "ngraph/ops/broadcast.hpp"
#include "ngraph/ops/dot.hpp"
#include "ngraph/ops/multiply.hpp"
#include "ngraph/ops/reshape.hpp"
#include "ngraph/ops/sum.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

//
// Helper function to compute the number of dot axes according to default behavior when
// they are not specified.
//
size_t default_reduction_axes_count(const std::shared_ptr<Node>& arg0,
                                    const std::shared_ptr<Node>& arg1)
{
    auto arg0_value_type = arg0->get_value_type();
    auto arg0_tensor_view_type = std::dynamic_pointer_cast<const TensorViewType>(arg0_value_type);
    if (nullptr == arg0_tensor_view_type)
    {
        throw ngraph_error("Dot arg0 does not have tensor view type");
    }
    auto arg0_shape = arg0_tensor_view_type->get_shape();

    auto arg1_value_type = arg1->get_value_type();
    auto arg1_tensor_view_type = std::dynamic_pointer_cast<const TensorViewType>(arg1_value_type);
    if (nullptr == arg1_tensor_view_type)
    {
        throw ngraph_error("Dot arg1 does not have tensor view type");
    }
    auto arg1_shape = arg1_tensor_view_type->get_shape();

    if (arg0_shape.size() == 0 || arg1_shape.size() == 0)
    {
        return 0;
    }
    else
    {
        return 1;
    }
}

op::Dot::Dot(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1)
    : Dot(arg0, arg1, default_reduction_axes_count(arg0, arg1))
{
}

op::Dot::Dot(const std::shared_ptr<Node>& arg0,
             const std::shared_ptr<Node>& arg1,
             size_t reduction_axes_count)
    : RequiresTensorViewArgs("Dot", {arg0, arg1})
    , m_reduction_axes_count(reduction_axes_count)
{
    auto arg0_tensor_type = get_inputs().at(0).get_tensor_view_type();
    auto arg1_tensor_type = get_inputs().at(1).get_tensor_view_type();

    if (arg0_tensor_type->get_element_type() != arg1_tensor_type->get_element_type())
    {
        throw ngraph_error("Arguments to dot must have the same element type");
    }

    vector<size_t> arg0_shape = arg0_tensor_type->get_shape();
    vector<size_t> arg1_shape = arg1_tensor_type->get_shape();

    if (reduction_axes_count > arg0_shape.size())
    {
        throw ngraph_error("Dot has too many axes for arg0");
    }

    if (reduction_axes_count > arg1_shape.size())
    {
        throw ngraph_error("Dot has too many axes for arg1");
    }

    for (size_t i = 0; i < reduction_axes_count; i++)
    {
        if (arg0_shape[arg0_shape.size() - reduction_axes_count + i] != arg1_shape[i])
        {
            throw ngraph_error("Dot axes do not have same length");
        }
    }

    vector<size_t> result_shape(arg0_shape.size() + arg1_shape.size() - 2 * reduction_axes_count);

    std::copy(arg0_shape.begin(), arg0_shape.end() - reduction_axes_count, result_shape.begin());
    std::copy(arg1_shape.begin() + reduction_axes_count,
              arg1_shape.end(),
              result_shape.begin() + (arg0_shape.size() - reduction_axes_count));

    auto result_type =
        make_shared<TensorViewType>(arg0_tensor_type->get_element_type(), result_shape);
    set_value_type_checked(result_type);
}

std::shared_ptr<op::Reshape> make_reshape_axes_to_front(const std::shared_ptr<Node>& n,
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

void op::Dot::generate_adjoints(autodiff::Adjoints& adjoints, const std::shared_ptr<Node>& delta)
{
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

    auto delta_reshaped = make_reshape_axes_to_front(delta, I_shape, K_shape);       // KI
    auto delta_reshaped_dot_y = make_shared<Dot>(y, delta_reshaped, K_shape.size()); // JI
    auto delta_reshaped_dot_y_reshaped =
        make_reshape_axes_to_front(delta_reshaped_dot_y, J_shape, I_shape); // IJ
    adjoints.add_delta(x, delta_reshaped_dot_y_reshaped);

    auto x_reshaped = make_reshape_axes_to_front(x, I_shape, J_shape);               // JI
    auto x_reshaped_dot_delta = make_shared<Dot>(x_reshaped, delta, I_shape.size()); // JK
    adjoints.add_delta(y, x_reshaped_dot_delta);
}
