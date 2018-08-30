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

#include "matmul_bias.hpp"
#include "ngraph/log.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

shared_ptr<Node> op::MatmulBias::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 2 && new_args.size() != 3)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<MatmulBias>(new_args.at(0),
                                   new_args.at(1),
                                   new_args.size() == 3 ? new_args.at(2) : nullptr,
                                   m_shape_w,
                                   m_shape_x,
                                   m_transpose_w,
                                   m_transpose_x,
                                   m_broadcast_axes);
}

op::MatmulBias::MatmulBias(shared_ptr<Node> W,
                           shared_ptr<Node> x,
                           shared_ptr<Node> b,
                           Shape shape_w,
                           Shape shape_x,
                           bool transpose_w,
                           bool transpose_x,
                           AxisSet axes)
    : RequiresTensorViewArgs("MatMulBias",
                             b == nullptr ? vector<shared_ptr<Node>>{W, x}
                                          : vector<shared_ptr<Node>>{W, x, b})
    , m_shape_w(shape_w)
    , m_shape_x(shape_x)
    , m_transpose_w(transpose_w)
    , m_transpose_x(transpose_x)
    , m_broadcast_axes(axes)

{
    if (axes.size() == 0 && b != nullptr)
    {
        throw ngraph_error("Bias but no broadcast axes");
    }

    if (b == nullptr && axes.size() != 0)
    {
        throw ngraph_error("Broadcast axes but no bias");
    }

    if (axes.size() > 2)
    {
        throw ngraph_error("Broadcasting to > 2D tensor");
    }

    if (shape_w.size() != 2)
    {
        NGRAPH_DEBUG << "W shape = " << vector_to_string(shape_w);
        throw ngraph_error("W.shape.rank != 2 while creating MatmulBias");
    }

    if (shape_x.size() != 2)
    {
        NGRAPH_DEBUG << "x shape = " << vector_to_string(shape_x);
        throw ngraph_error("x.shape.rank != 2 while creating MatmulBias");
    }

    size_t dot_dimension_w = (transpose_w) ? 0 : 1;
    size_t dot_dimension_x = (transpose_x) ? 1 : 0;

    NGRAPH_DEBUG << "dot_dimension_w = " << dot_dimension_w
                 << " , dot_dimension_x = " << dot_dimension_x;
    NGRAPH_DEBUG << "W shape = " << vector_to_string(shape_w)
                 << " , x shape = " << vector_to_string(shape_x);

    if (shape_w.at(dot_dimension_w) != shape_x.at(dot_dimension_x))
    {
        throw ngraph_error("product dimensions are not equal while creating MatmulBias");
    }

    Shape dot_shape{shape_w.at(1 - dot_dimension_w), shape_x.at(1 - dot_dimension_x)};
    NGRAPH_DEBUG << "dot_shape shape = " << vector_to_string(dot_shape);

    if (b)
    {
        NGRAPH_DEBUG << "b shape = " << vector_to_string(b->get_shape());
    }

    add_output(W->get_element_type(), dot_shape);
}
