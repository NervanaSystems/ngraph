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
#include "ngraph/dimension.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;


op::BatchDot::BatchDot(const shared_ptr<Node>& a, const shared_ptr<Node>& b, bool transpose_a, bool transpose_b)
    : Op("BatchDot", check_single_output_args({a, b}))
    , m_transpose_a(transpose_a)
    , m_transpose_b(transpose_b)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::BatchDot::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<BatchDot>(new_args.at(0), new_args.at(1), m_transpose_a, m_transpose_b);
}

void op::BatchDot::validate_and_infer_types()
{
  
    // check input types
    auto a_et = get_input_element_type(0);
    auto b_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          a_et.compatible(b_et),
                          "Inputs a and b must have compatible element type.");
    // check input shapes 
    const PartialShape& a_shape = get_input_partial_shape(0);
    const PartialShape& b_shape = get_input_partial_shape(1);
    NODE_VALIDATION_CHECK(this,
                          a_shape.rank().compatible(3),
                          "Input a shape must have rank 3, got ",
                          a_shape.rank(),
                          ".");
    NODE_VALIDATION_CHECK(this,
                          b_shape.rank().compatible(3),
                          "Input b shape must have rank 3, got ",
                          b_shape.rank(),
                          ".");
    
    size_t dot_dim_a = (m_transpose_a) ? 1 : 2;
    size_t dot_dim_b = (m_transpose_b) ? 2 : 1;

    PartialShape output_shape(PartialShape::dynamic(3));
    if (a_shape.rank().is_static() && b_shape.rank().is_static() &&
        a_shape.rank().compatible(3) && b_shape.rank().compatible(3)) {
        NODE_VALIDATION_CHECK(this,
                              a_shape[dot_dim_a].compatible(b_shape[dot_dim_b]),
                              "Product dimensions are not equal while creating BatchDot.");
        output_shape = PartialShape{a_shape[0], a_shape[3-dot_dim_a], b_shape[3-dot_dim_b]};
    }

    set_output_type(0, get_input_element_type(0), output_shape);
}

void op::BatchDot::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    if (get_input_partial_shape(0).is_static() 
        && get_input_partial_shape(1).is_static()) {
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
    else {
        throw ngraph_error("generate_adjoints not implemented for BatchDot with dynamic input shapes");
    }
}
