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

#include "ngraph/builder/matmul_factory.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/frontend/fluid/operators/matmul.hpp"
#include "ngraph/op/reshape.hpp"

#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/experimental/batch_mat_mul.hpp"
#include "ngraph/op/sum.hpp"

using namespace std;
using namespace ngraph::fluid;

constexpr NodeTypeInfo MatMul::type_info;
MatMul::MatMul(const Output<Node>& A,
               const Output<Node>& B,
               const bool& transpose_a,
               const bool& transpose_b)
    : FusedOp(OutputVector{A, B})
    , m_transpose_a{transpose_a}
    , m_transpose_b{transpose_b}
{
    constructor_validate_and_infer_types();
}

template <class Input>
void DecomposeLogic(Input& input, bool transpose, bool reverse = false)
{
    auto rank = input.get_shape().size();

    if (rank < 2)
    {
        if (rank)
        {
            if (reverse)
            {
                input =
                    make_shared<op::Reshape>(input, AxisVector{0}, Shape{input.get_shape()[0], 1});
            }
            else
            {
                input =
                    make_shared<op::Reshape>(input, AxisVector{0}, Shape{1, input.get_shape()[0]});
            }
        }
        else
        {
            input = make_shared<op::Reshape>(input, AxisVector{}, Shape{1, 1});
        }
        rank = 2;
    }

    if (transpose)
    {
        vector<size_t> axes_order(rank);
        iota(axes_order.begin(), axes_order.end(), 0);
        swap(axes_order[rank - 1], axes_order[rank - 2]);
        input = builder::reorder_axes(input, axes_order);
    }
}

inline NodeVector remove_1(std::shared_ptr<ngraph::Node> input_node)
{
    auto input_shape = input_node->get_shape();
    AxisVector axis(input_shape.size());
    iota(axis.begin(), axis.end(), 0);
    Shape shape(input_shape.begin(), input_shape.end());

    auto b_remove = std::remove(shape.begin(), shape.end(), 1);
    shape.erase(b_remove, shape.end());
    Output<Node> node(input_node);

    auto reshape = make_shared<op::Reshape>(node, axis, shape);
    NodeVector final_vector{reshape};

    return final_vector;
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
    auto A = input_value(0);
    auto B = input_value(1);

    DecomposeLogic(A, m_transpose_a);
    DecomposeLogic(B, m_transpose_b, true);
    builder::MatmulFactory factory({A, B});

    auto node_vector_matmul = factory.make_matmul_op();
    auto first_item_node_vector = node_vector_matmul[0];
    auto b = first_item_node_vector->get_shape().begin();
    auto e = first_item_node_vector->get_shape().end();
    auto it = std::find(b, e, 1);

    if (it != e)
    {
        node_vector_matmul = remove_1(first_item_node_vector);
    }

    return node_vector_matmul;
}

shared_ptr<Node> MatMul::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<MatMul>(new_args.at(0), new_args.at(1), m_transpose_a, m_transpose_b);
}

constexpr NodeTypeInfo MatMulBackward::type_info;
MatMulBackward::MatMulBackward(const Output<Node>& A,
                               const Output<Node>& B,
                               const Output<Node>& Out,
                               bool is_dx,
                               bool is_dy,
                               const bool& transpose_a,
                               const bool& transpose_b)
    : FusedOp(OutputVector{A, B, Out})
    , is_x(is_x)
    , is_y(is_y)
    , m_transpose_a{transpose_a}
    , m_transpose_b{transpose_b}
{
    constructor_validate_and_infer_types();
}

std::shared_ptr<ngraph::Node>
    MatMulBackward::broadcast3D(const std::shared_ptr<ngraph::Node>& input, size_t axis0) const
{
    auto shape = input->get_shape();
    size_t n = shape.size();
    if (n == 2)
    {
        auto output = std::make_shared<ngraph::op::Broadcast>(
            input, ngraph::Shape{axis0, shape[0], shape[1]}, ngraph::AxisSet{0});
        return output;
    }
    return input;
}

std::shared_ptr<ngraph::Node> MatMulBackward::transposeAndFlat3D(
    const std::shared_ptr<ngraph::Node>& input, const bool transpose, bool x) const
{
    auto shape = input->get_shape();
    size_t n = shape.size();
    std::shared_ptr<ngraph::Node> output;
    if (n >= 3)
    {
        std::vector<size_t> order(n);
        std::iota(std::begin(order), std::end(order), 0);
        size_t outer = 1;
        for (size_t i = 0; i < n - 2; i++)
        {
            outer = outer * shape[i];
        }
        std::vector<size_t> reshape{outer, shape[n - 2], shape[n - 1]};

        if (transpose == true)
        {
            order[n - 2] = n - 1;
            order[n - 1] = n - 2;
            reshape[2] = shape[n - 2];
            reshape[1] = shape[n - 1];
        }
        output = std::make_shared<ngraph::op::Reshape>(
            input, ngraph::AxisVector(order), ngraph::Shape(reshape));
    }
    else
    {
        std::shared_ptr<ngraph::Node> temp;
        if (n == 1 && x == true)
        {
            temp = std::make_shared<ngraph::op::Reshape>(
                input, ngraph::AxisVector{0}, ngraph::Shape{1, shape[0]});
        }
        else if (n == 1 && x == false)
        {
            temp = std::make_shared<ngraph::op::Reshape>(
                input, ngraph::AxisVector{0}, ngraph::Shape{shape[0], 1});
        }
        else
        {
            temp = input;
        }
        auto temp_shape = temp->get_shape();
        if (transpose == true)
        {
            output = std::make_shared<ngraph::op::Reshape>(
                temp, ngraph::AxisVector{1, 0}, ngraph::Shape{temp_shape[1], temp_shape[0]});
        }
        else
        {
            output = temp;
        }
    }
    return output;
}

std::shared_ptr<ngraph::Node> MatMulBackward::dotOp(const std::shared_ptr<ngraph::Node>& a,
                                                    const std::shared_ptr<ngraph::Node>& b) const
{
    std::shared_ptr<ngraph::Node> out;
    auto a_shape = a->get_shape();
    auto na = a_shape.size();
    auto b_shape = b->get_shape();
    auto nb = b_shape.size();
    if (na > 2 && nb > 2)
    {
        out = std::make_shared<op::BatchMatMul>(a, b);
    }
    else
    {
        out = std::make_shared<op::Dot>(a, b);
    }
    return out;
}

std::shared_ptr<ngraph::Node> MatMulBackward::reshapeToOriginal(std::shared_ptr<ngraph::Node> input,
                                                                const ngraph::Shape& shape) const
{
    auto input_shape = input->get_shape();
    std::vector<size_t> axis(input_shape.size());
    std::iota(axis.begin(), axis.end(), 0);
    auto out = std::make_shared<ngraph::op::Reshape>(input, axis, shape);
    return out;
}

NodeVector MatMulBackward::decompose_op() const
{
    auto x = input_value(0).get_node_shared_ptr();
    auto y = input_value(1).get_node_shared_ptr();
    auto dout = input_value(2).get_node_shared_ptr();

    //  auto& dout = OutGrad;
    //  auto& x = m_A;
    //  auto& y = m_B;
    auto dout_shape = dout->get_shape();
    auto x_shape = x->get_shape();
    auto y_shape = y->get_shape();
    size_t nx = x_shape.size();
    size_t ny = y_shape.size();
    size_t ndout = dout_shape.size();
    std::shared_ptr<ngraph::Node> x2, y2;
    std::shared_ptr<ngraph::Node> dout2;

    x2 = transposeAndFlat3D(x, false);
    y2 = transposeAndFlat3D(y, false, false);
    dout2 = transposeAndFlat3D(dout, false);
    auto x2_shape = x2->get_shape();
    auto y2_shape = y2->get_shape();
    if (nx >= 3 || ny >= 3)
    {
        std::shared_ptr<ngraph::Node> dout_temp;
        if (ndout == 2)
        {
            dout_temp = std::make_shared<ngraph::op::Reshape>(
                dout, ngraph::AxisVector{0, 1}, ngraph::Shape{dout_shape[0], dout_shape[1], 1});
            if (ny < 3)
            {
                dout2 = dout_temp;
            }
            else
            {
                dout2 = transposeAndFlat3D(dout_temp, true);
            }
        }
        x2 = broadcast3D(x2, y_shape[0]);
        y2 = broadcast3D(y2, x_shape[0]);
    }
    else
    {
        dout2 = transposeAndFlat3D(dout, false, nx == 1 && m_transpose_a == false);
    }

    if (m_transpose_b == false)
    {
        y2 = transposeAndFlat3D(y2, true);
    }
    if (m_transpose_a == false)
    {
        x2 = transposeAndFlat3D(x2, true);
    }
    auto dx = dotOp(dout2, y2);
    auto dy = dotOp(x2, dout2);
    if (m_transpose_a == true)
    {
        dx = transposeAndFlat3D(dx, true);
    }
    if (m_transpose_b == true)
    {
        dy = transposeAndFlat3D(dy, true);
    }

    if (nx < 3 && ny >= 3)
    {
        dx = std::make_shared<ngraph::op::Sum>(dx, ngraph::AxisSet{0});
    }
    if (ny < 3 && nx >= 3)
    {
        dy = std::make_shared<ngraph::op::Sum>(dy, ngraph::AxisSet{0});
    }

    auto dx_t = reshapeToOriginal(dx, x_shape);
    auto dy_t = reshapeToOriginal(dy, y_shape);

    return NodeVector{dx_t, dy_t, dout};
}

shared_ptr<Node> MatMulBackward::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<MatMulBackward>(
        new_args.at(0), new_args.at(1), new_args.at(2), is_x, is_y, m_transpose_a, m_transpose_b);
}

void MatMulBackward::pre_validate_and_infer_types()
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