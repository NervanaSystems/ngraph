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

#include "matmul.hpp"
#include "ngraph/builder/matmul_factory.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/reshape.hpp"

using namespace std;
using namespace ngraph;

const string op::MatMul::type_name{"MatMul"};

op::MatMul::MatMul(const shared_ptr<Node>& A,
                   const shared_ptr<Node>& B,
                   const int& transpose_a,
                   const int& transpose_b)
    : FusedOp(NodeVector{A, B})
    , m_transpose_a{transpose_a}
    , m_transpose_b{transpose_b}
{
    constructor_validate_and_infer_types();
}

NodeVector op::MatMul::decompose_op() const
{
    auto A = get_argument(0);
    auto B = get_argument(1);

    int a_rank = A->get_shape().size();
    if (a_rank < 2)
    {
        A = a_rank == 0 ? make_shared<op::Reshape>(A, AxisVector{}, Shape{1, 1})
                        : make_shared<op::Reshape>(A, AxisVector{1}, Shape{1, A->get_shape()[0]});
        a_rank = 2;
    }

    int b_rank = B->get_shape().size();
    if (b_rank < 2)
    {
        B = b_rank == 0 ? make_shared<op::Reshape>(B, AxisVector{}, Shape{1, 1})
                        : make_shared<op::Reshape>(B, AxisVector{1}, Shape{1, B->get_shape()[0]});
        b_rank = 2;
    }

    if (m_transpose_a)
    {
        vector<size_t> axes_order(a_rank);
        // generate default axes_order.
        iota(axes_order.begin(), axes_order.end(), 0);
        // transpose the last 2 spatial dims
        swap(axes_order[a_rank - 1], axes_order[a_rank - 2]);
        A = builder::reorder_axes(A, axes_order);
    }

    if (m_transpose_b)
    {
        vector<size_t> axes_order(b_rank);
        iota(axes_order.begin(), axes_order.end(), 0);
        swap(axes_order[b_rank - 1], axes_order[b_rank - 2]);
        B = builder::reorder_axes(B, axes_order);
    }

    auto factory = builder::MatmulFactory({A, B});
    return factory.make_matmul_op();
}

shared_ptr<Node> op::MatMul::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<MatMul>(new_args.at(0), new_args.at(1), m_transpose_a, m_transpose_b);
}
