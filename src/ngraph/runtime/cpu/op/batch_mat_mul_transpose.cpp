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

#include "batch_mat_mul_transpose.hpp"
#include "ngraph/dimension.hpp"
#include "ngraph/log.hpp"
#include "ngraph/op/experimental/batch_mat_mul.hpp"
#include "ngraph/op/experimental/dyn_reshape.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::BatchMatMulTranspose::type_info;

op::BatchMatMulTranspose::BatchMatMulTranspose(const Output<Node>& arg0,
                                               const Output<Node>& arg1,
                                               bool transpose_arg0,
                                               bool transpose_arg1)
    : Op({arg0, arg1})
    , m_transpose_arg0(transpose_arg0)
    , m_transpose_arg1(transpose_arg1)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::BatchMatMulTranspose::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<BatchMatMulTranspose>(
        new_args.at(0), new_args.at(1), m_transpose_arg0, m_transpose_arg1);
}

void op::BatchMatMulTranspose::validate_and_infer_types()
{
    // Check input types
    const auto& arg0_et = get_input_element_type(0);
    const auto& arg1_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          arg0_et.compatible(arg1_et),
                          "Inputs arg0 and arg1 must have compatible element type.");
    // Check input shapes
    const PartialShape& arg0_shape = get_input_partial_shape(0);
    const PartialShape& arg1_shape = get_input_partial_shape(1);
    NODE_VALIDATION_CHECK(this,
                          arg0_shape.rank().compatible(3),
                          "Input arg0 shape must have rank 3, got ",
                          arg0_shape.rank(),
                          ".");
    NODE_VALIDATION_CHECK(this,
                          arg1_shape.rank().compatible(3),
                          "Input arg1 shape must have rank 3, got ",
                          arg1_shape.rank(),
                          ".");

    size_t dot_dim_arg0 = (m_transpose_arg0) ? 1 : 2;
    size_t dot_dim_arg1 = (m_transpose_arg1) ? 2 : 1;
    // We expect output shape always have rank 3
    PartialShape output_shape(PartialShape::dynamic(3));

    // Construct output shape with more information if avalible.
    if (arg0_shape.rank().same_scheme(3) && arg1_shape.rank().same_scheme(3))
    {
        NODE_VALIDATION_CHECK(
            this,
            arg0_shape[0].compatible(arg1_shape[0]),
            "Batch size dimensions are not equal while creating BatchMatMulTranspose.");
        NODE_VALIDATION_CHECK(
            this,
            arg0_shape[dot_dim_arg0].compatible(arg1_shape[dot_dim_arg1]),
            "Product dimensions are not equal while creating BatchMatMulTranspose.");
        auto batch_size = arg0_shape[0].is_static() ? arg0_shape[0] : arg1_shape[0];
        output_shape =
            PartialShape{batch_size, arg0_shape[3 - dot_dim_arg0], arg1_shape[3 - dot_dim_arg1]};
    }
    auto output_et = arg0_et.is_dynamic() ? arg1_et : arg0_et;
    set_output_type(0, output_et, output_shape);
}

void op::BatchMatMulTranspose::generate_adjoints(autodiff::Adjoints& adjoints,
                                                 const NodeVector& deltas)
{
    auto delta = deltas.at(0); // NxIxK

    auto arg0 = input(0).get_source_output().get_node_shared_ptr(); // NxIxJ (maybe transposed)
    auto arg1 = input(1).get_source_output().get_node_shared_ptr(); // NxJxK (maybe transposed)

    // If arg1 is already transposed, it does not need to be transposed again
    auto delta_dot_arg1 =
        make_shared<op::BatchMatMulTranspose>(delta, arg1, false, !m_transpose_arg1); // IK.KJ->IJ
    // If arg0 is transposed, the result need to be transposed to match original arg0 shape.
    if (m_transpose_arg0)
    {
        adjoints.add_delta(arg0, util::batch_mat_transpose(delta_dot_arg1));
    }
    else
    {
        adjoints.add_delta(arg0, delta_dot_arg1);
    }

    auto arg0_dot_delta =
        make_shared<BatchMatMulTranspose>(arg0, delta, !m_transpose_arg0, false); // JI.IK->JK
    if (m_transpose_arg1)
    {
        adjoints.add_delta(arg1, util::batch_mat_transpose(arg0_dot_delta));
    }
    else
    {
        adjoints.add_delta(arg1, arg0_dot_delta);
    }
}
