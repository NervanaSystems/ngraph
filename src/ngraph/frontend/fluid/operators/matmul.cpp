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
