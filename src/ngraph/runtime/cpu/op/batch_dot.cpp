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

#include "batch_dot.hpp"
#include "ngraph/log.hpp"
#include "ngraph/util.hpp"
#include "ngraph/op/reshape.hpp"

using namespace std;
using namespace ngraph;

shared_ptr<Node> op::BatchDot::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<BatchDot>(new_args.at(0), new_args.at(1), m_transpose_a, m_transpose_b);
}

op::BatchDot::BatchDot(shared_ptr<Node> a, shared_ptr<Node> b, bool transpose_a, bool transpose_b)
    : Op("BatchDot", check_single_output_args({a, b}))
    , m_transpose_a(transpose_a)
    , m_transpose_b(transpose_b)
{
    constructor_validate_and_infer_types();

    const auto& shape_a = a->get_shape();
    const auto& shape_b = b->get_shape();
    if (shape_a.size() != 3 || shape_b.size() != 3)
    {
        NGRAPH_DEBUG << "shape_a = " << vector_to_string(shape_a);
        NGRAPH_DEBUG << "shape_b = " << vector_to_string(shape_b);
        throw ngraph_error("shape rank != 3 while creating BatchDot");
    }
    if (a->get_element_type() != b->get_element_type())
    {
        throw ngraph_error("input element types did not match while creating BatchDot");
    }
    size_t dot_dimension_a = (transpose_a) ? 1 : 2;
    size_t dot_dimension_b = (transpose_b) ? 2 : 1;

    NGRAPH_DEBUG << "dot_dimension_a = " << dot_dimension_a
                 << " , dot_dimension_b = " << dot_dimension_b;
    NGRAPH_DEBUG << "a shape = " << vector_to_string(shape_a)
                 << " , b shape = " << vector_to_string(shape_b);

    if (shape_a.at(dot_dimension_a) != shape_b.at(dot_dimension_b))
    {
        throw ngraph_error("product dimensions are not equal while creating BatchDot");
    }

    Shape dot_shape{
        shape_a.at(0), shape_a.at(3 - dot_dimension_a), shape_b.at(3 - dot_dimension_b)};
    NGRAPH_DEBUG << "dot_shape shape = " << vector_to_string(dot_shape);

    set_output_type(0, a->get_element_type(), dot_shape);
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

void op::BatchDot::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
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
    I_shape.insert(I_shape.begin(), x_shape.begin(), x_shape.end() - 1);
    J_shape.insert(J_shape.begin(), y_shape.begin(), y_shape.begin() + 1);
    K_shape.insert(K_shape.begin(), y_shape.begin() + J_shape.size(), y_shape.end());

    auto y_reshaped = make_reshape_axes_to_front(y, J_shape, K_shape);               // KJ
    auto delta_dot_y_reshaped = make_shared<op::BatchDot>(delta, y_reshaped, false, m_transpose_b); // IK.KJ->IJ
    adjoints.add_delta(x, delta_dot_y_reshaped);

    auto x_reshaped = make_reshape_axes_to_front(x, I_shape, J_shape);               // JI
    auto x_reshaped_dot_delta = make_shared<BatchDot>(x_reshaped, delta, m_transpose_a, false); // JI.IK->JK
    adjoints.add_delta(y, x_reshaped_dot_delta);
}
