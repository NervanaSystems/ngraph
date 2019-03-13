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
#include "ngraph/op/reshape.hpp"
#include "ngraph/util.hpp"

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

op::BatchDot::BatchDot(const NodeOutput& a, const NodeOutput& b, bool transpose_a, bool transpose_b)
    : Op("BatchDot", {a, b})
    , m_transpose_a(transpose_a)
    , m_transpose_b(transpose_b)
{
    constructor_validate_and_infer_types();

    const auto& shape_a = a.get_shape();
    const auto& shape_b = b.get_shape();
    if (shape_a.size() != 3 || shape_b.size() != 3)
    {
        NGRAPH_DEBUG << "shape_a = " << vector_to_string(shape_a);
        NGRAPH_DEBUG << "shape_b = " << vector_to_string(shape_b);
        throw ngraph_error("shape rank != 3 while creating BatchDot");
    }
    if (a.get_element_type() != b.get_element_type())
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

    set_output_type(0, a.get_element_type(), dot_shape);
}

void op::BatchDot::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0); // NxIxK

    auto a = get_inputs().at(0).get_output().get_node(); // NxIxJ (maybe transposed)
    auto b = get_inputs().at(1).get_output().get_node(); // NxJxK (maybe transposed)

    auto batch_transpose = [](const shared_ptr<Node>& node) {
        const auto& batch_shape = node->get_shape();
        // index 0 is the batch, only transposing the others.
        AxisVector input_order{0, 2, 1};
        Shape output_shape{batch_shape[0], batch_shape[2], batch_shape[1]};
        return make_shared<op::Reshape>(node, input_order, output_shape);
    };

    // if b is already transposed, it does not need to be transposed again
    auto delta_dot_b = make_shared<op::BatchDot>(delta, b, false, !m_transpose_b); // IK.KJ->IJ
    // if a is transposed, the result need to be transposed to match original a shape.
    if (m_transpose_a)
    {
        adjoints.add_delta(a, batch_transpose(delta_dot_b));
    }
    else
    {
        adjoints.add_delta(a, delta_dot_b);
    }

    auto a_dot_delta = make_shared<BatchDot>(a, delta, !m_transpose_a, false); // JI.IK->JK
    if (m_transpose_b)
    {
        adjoints.add_delta(b, batch_transpose(a_dot_delta));
    }
    else
    {
        adjoints.add_delta(b, a_dot_delta);
    }
}
