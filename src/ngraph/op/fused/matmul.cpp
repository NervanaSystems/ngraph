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
#include <memory>
#include <numeric>

#include "matmul.hpp"
#include "ngraph/builder/matmul_factory.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/reshape.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::MatMul::type_info;

op::MatMul::MatMul(const Output<Node>& A,
                   const Output<Node>& B,
                   const bool& transpose_a,
                   const bool& transpose_b)
    : FusedOp(OutputVector{A, B})
    , m_transpose_a{transpose_a}
    , m_transpose_b{transpose_b}
{
    constructor_validate_and_infer_types();
}

void op::MatMul::pre_validate_and_infer_types()
{
    element::Type result_et;
    NODE_VALIDATION_CHECK(
        this,
        element::Type::merge(result_et, get_input_element_type(0), get_input_element_type(1)),
        "Arguments do not have the same element type (arg0 element type: ",
        get_input_element_type(0),
        ", arg1 element type: ",
        get_input_element_type(1),
        ").");

    const Rank& A_rank = get_input_partial_shape(0).rank();
    const Rank& B_rank = get_input_partial_shape(1).rank();

    if (A_rank.is_static() && B_rank.is_static())
    {
        Rank max_rank = int64_t(A_rank) > int64_t(B_rank) ? A_rank : B_rank;
        set_output_type(0, result_et, PartialShape::dynamic(max_rank));
    }
}

NodeVector op::MatMul::decompose_op() const
{
    auto A = input_value(0);
    auto B = input_value(1);

    const auto a_rank = A.get_shape().size();
    const auto b_rank = B.get_shape().size();

    if (m_transpose_a && a_rank >= 2)
    {
        vector<size_t> axes_order(a_rank);
        // generate default axes_order.
        iota(axes_order.begin(), axes_order.end(), 0);
        // transpose the last 2 spatial dims
        swap(axes_order[a_rank - 1], axes_order[a_rank - 2]);
        A = builder::reorder_axes(A, axes_order);
    }

    if (m_transpose_b && b_rank >= 2)
    {
        vector<size_t> axes_order(b_rank);
        iota(axes_order.begin(), axes_order.end(), 0);
        swap(axes_order[b_rank - 1], axes_order[b_rank - 2]);
        B = builder::reorder_axes(B, axes_order);
    }

    builder::MatmulFactory factory({A, B});
    return factory.make_matmul_op();
}

shared_ptr<Node> op::MatMul::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<MatMul>(new_args.at(0), new_args.at(1), m_transpose_a, m_transpose_b);
}
