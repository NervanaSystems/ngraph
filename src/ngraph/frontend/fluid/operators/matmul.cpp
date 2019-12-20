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
#include <memory>
#include <numeric>

#include "ngraph/frontend/fluid/operators/matmul.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/experimental/batch_mat_mul.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph::fluid;

shared_ptr<Node> broadcast_to_3d(const shared_ptr<Node>& input, size_t axis0)
{
    auto shape = input->get_shape();
    size_t n = shape.size();

    if (n == 2)
    {
        auto output =
            make_shared<op::Broadcast>(input, Shape{axis0, shape[0], shape[1]}, AxisSet{0});
        return output;
    }

    return input;
}

shared_ptr<Node> transpose_and_flatten3d(const shared_ptr<Node>& input,
                                         const bool transpose,
                                         const bool x = true)
{
    auto shape = input->get_shape();
    size_t n = shape.size();
    shared_ptr<Node> output;

    if (n >= 3)
    {
        vector<size_t> order(n);
        iota(begin(order), end(order), 0);
        size_t outer = 1;
        for (size_t i = 0; i < n - 2; i++)
        {
            outer = outer * shape[i];
        }
        vector<size_t> reshape{outer, shape[n - 2], shape[n - 1]};

        if (transpose)
        {
            order[n - 2] = n - 1;
            order[n - 1] = n - 2;
            reshape[2] = shape[n - 2];
            reshape[1] = shape[n - 1];
        }
        output = make_shared<op::Reshape>(input, AxisVector(order), Shape(reshape));
    }
    else
    {
        shared_ptr<Node> temp;
        if (n == 1 && x == true)
        {
            temp = make_shared<op::Reshape>(input, AxisVector{0}, Shape{1, shape[0]});
        }
        else if (n == 1 && x == false)
        {
            temp = make_shared<op::Reshape>(input, AxisVector{0}, Shape{shape[0], 1});
        }
        else
        {
            temp = input;
        }
        auto temp_shape = temp->get_shape();
        if (transpose == true)
        {
            output = make_shared<op::Reshape>(
                temp, AxisVector{1, 0}, Shape{temp_shape[1], temp_shape[0]});
        }
        else
        {
            output = temp;
        }
    }

    return output;
}

shared_ptr<Node> dot_helper(const shared_ptr<Node>& a, const shared_ptr<Node>& b)
{
    shared_ptr<Node> out;

    if (a->get_shape().size() > 2 && b->get_shape().size() > 2)
    {
        out = make_shared<op::BatchMatMul>(a, b);
    }
    else
    {
        out = make_shared<op::Dot>(a, b);
    }

    return out;
}

shared_ptr<Node> reshape_to_original(shared_ptr<Node> input, const Shape& shape)
{
    auto input_shape = input->get_shape();
    return make_shared<op::Reshape>(input, get_default_order(input_shape), shape);
}

constexpr NodeTypeInfo MatMul::type_info;
MatMul::MatMul(const Output<Node>& A,
               const Output<Node>& B,
               const bool transpose_a,
               const bool transpose_b)
    : FusedOp(OutputVector{A, B})
    , m_transpose_a{transpose_a}
    , m_transpose_b{transpose_b}
{
    constructor_validate_and_infer_types();
}

void MatMul::pre_validate_and_infer_types()
{
    element::Type input_element_type = get_input_element_type(0);
    PartialShape pshape_A = get_input_partial_shape(0);
    PartialShape pshape_B = get_input_partial_shape(1);

    NODE_VALIDATION_CHECK(this,
                          input_element_type.is_dynamic() || input_element_type.is_real(),
                          "Argument element type must be f16, bf16, f32, f64 or dynamic (got ",
                          input_element_type,
                          ").");

    if (pshape_A.is_dynamic() || pshape_B.is_dynamic())
    {
        set_output_type(0, input_element_type, PartialShape::dynamic());
    }
}

NodeVector MatMul::decompose_op() const
{
    auto x = input_value(0).get_node_shared_ptr();
    auto y = input_value(1).get_node_shared_ptr();

    auto x_shape = x->get_shape();
    auto y_shape = y->get_shape();

    size_t nx = x_shape.size();
    size_t ny = y_shape.size();

    x = transpose_and_flatten3d(x, m_transpose_a, true);
    y = transpose_and_flatten3d(y, m_transpose_b, false);

    auto y_shape3 = y->get_shape();
    auto x_shape3 = x->get_shape();

    shared_ptr<Node> out;
    Shape out_shape;

    if (nx > 2 || ny > 2)
    {
        Shape out_shape = x_shape;
        if (nx != 3)
        {
            x = broadcast_to_3d(x, y_shape3[0]);
            out_shape = y_shape;
        }
        if (ny != 3)
        {
            y = broadcast_to_3d(y, x_shape3[0]);
            out_shape = x_shape;
        }
        auto nout = out_shape.size();
        auto out3 = make_shared<op::BatchMatMul>(x, y);
        auto out3_shape = out3->get_shape();
        out_shape[nout - 1] = out3_shape[2];
        out_shape[nout - 2] = out3_shape[1];
        out = make_shared<op::Reshape>(out3, AxisVector{0, 1, 2}, out_shape);
    }
    else
    {
        out = make_shared<op::Dot>(x, y);
    }

    out_shape = out->get_shape();
    auto axis_vector = get_default_order(out_shape);

    for (size_t i = out_shape.size() - 1; i > 0; i--)
    {
        if (out_shape[i] == 1)
        {
            out_shape.erase(out_shape.begin() + i);
        }
    }

    auto out_reshaped = make_shared<op::Reshape>(out, axis_vector, out_shape);

    return {out_reshaped};
}

shared_ptr<Node> MatMul::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<MatMul>(new_args.at(0), new_args.at(1), m_transpose_a, m_transpose_b);
}

constexpr NodeTypeInfo MatMulGrad::type_info;
MatMulGrad::MatMulGrad(const Output<Node>& A,
                       const Output<Node>& B,
                       const Output<Node>& Out,
                       const bool transpose_a,
                       const bool transpose_b)
    : FusedOp(OutputVector{A, B, Out})
    , m_transpose_a{transpose_a}
    , m_transpose_b{transpose_b}
{
    constructor_validate_and_infer_types();
}

void MatMulGrad::pre_validate_and_infer_types()
{
    element::Type input_element_type = get_input_element_type(0);

    NODE_VALIDATION_CHECK(this,
                          input_element_type.is_dynamic() || input_element_type.is_real(),
                          "Argument element type must be f16, bf16, f32, f64 or dynamic (got ",
                          input_element_type,
                          ").");

    if (get_input_partial_shape(0).is_dynamic() || get_input_partial_shape(1).is_dynamic() ||
        get_input_partial_shape(2).is_dynamic())
    {
        set_output_type(0, input_element_type, PartialShape::dynamic());
        set_output_type(1, input_element_type, PartialShape::dynamic());
    }
}
NodeVector MatMulGrad::decompose_op() const
{
    auto x = input_value(0).get_node_shared_ptr();
    auto y = input_value(1).get_node_shared_ptr();
    auto dout = input_value(2).get_node_shared_ptr();

    auto dout_shape = dout->get_shape();
    auto x_shape = x->get_shape();
    auto y_shape = y->get_shape();
    size_t nx = x_shape.size();
    size_t ny = y_shape.size();
    size_t ndout = dout_shape.size();

    shared_ptr<Node> x2, y2, dout2;

    x2 = transpose_and_flatten3d(x, false);
    y2 = transpose_and_flatten3d(y, false, false);
    dout2 = transpose_and_flatten3d(dout, false);

    auto x2_shape = x2->get_shape();
    auto y2_shape = y2->get_shape();

    if (nx >= 3 || ny >= 3)
    {
        shared_ptr<Node> dout_temp;
        if (ndout == 2)
        {
            dout_temp = make_shared<op::Reshape>(
                dout, AxisVector{0, 1}, Shape{dout_shape[0], dout_shape[1], 1});
            if (ny < 3)
            {
                dout2 = dout_temp;
            }
            else
            {
                dout2 = transpose_and_flatten3d(dout_temp, true);
            }
        }
        x2 = broadcast_to_3d(x2, y_shape[0]);
        y2 = broadcast_to_3d(y2, x_shape[0]);
    }
    else
    {
        dout2 = transpose_and_flatten3d(dout, false, nx == 1 && m_transpose_a == false);
    }

    if (m_transpose_b == false)
    {
        y2 = transpose_and_flatten3d(y2, true);
    }
    if (m_transpose_a == false)
    {
        x2 = transpose_and_flatten3d(x2, true);
    }

    auto dx = dot_helper(dout2, y2);
    auto dy = dot_helper(x2, dout2);

    if (m_transpose_a == true)
    {
        dx = transpose_and_flatten3d(dx, true);
    }
    if (m_transpose_b == true)
    {
        dy = transpose_and_flatten3d(dy, true);
    }

    if (nx < 3 && ny >= 3)
    {
        dx = make_shared<op::Sum>(dx, AxisSet{0});
    }
    if (ny < 3 && nx >= 3)
    {
        dy = make_shared<op::Sum>(dy, AxisSet{0});
    }

    auto dx_t = reshape_to_original(dx, x_shape);
    auto dy_t = reshape_to_original(dy, y_shape);

    return NodeVector{dx_t, dy_t};
}

shared_ptr<Node> MatMulGrad::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<MatMulGrad>(
        new_args.at(0), new_args.at(1), new_args.at(2), m_transpose_a, m_transpose_b);
}
