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

#include "ngraph/op/lrn.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/multiply.hpp"

using namespace std;
using namespace ngraph;

const string op::LRN::type_name{"LRN"};

op::LRN::LRN(const Output<Node>& arg, double alpha, double beta, double bias, size_t size)
    : LRN(arg, op::Constant::create(element::i32, Shape{1}, {1}), alpha, beta, bias, size)
{
}

op::LRN::LRN(const Output<Node>& arg,
             const Output<Node>& axes,
             double alpha,
             double beta,
             double bias,
             size_t size)
    : Op({arg, axes})
    , m_alpha(alpha)
    , m_beta(beta)
    , m_bias(bias)
    , m_size(size)
{
    constructor_validate_and_infer_types();
}

void op::LRN::validate_and_infer_types()
{
    element::Type arg_type = get_input_element_type(0);
    PartialShape arg_shape = get_input_partial_shape(0);
    set_output_type(0, arg_type, arg_shape);

    const PartialShape& input_shape = get_input_partial_shape(0);
    const PartialShape& axes_shape = get_input_partial_shape(1);

    NODE_VALIDATION_CHECK(this,
                          input_shape.rank().is_dynamic() ||
                              static_cast<size_t>(input_shape.rank()) >= 3,
                          "Argument must have rank >= 3 (argument shape: ",
                          input_shape,
                          ").");

    NODE_VALIDATION_CHECK(this, axes_shape.is_static(), "Input axes must be static.");

    NODE_VALIDATION_CHECK(this,
                          static_cast<size_t>(axes_shape.rank()) == 1,
                          "Input axes must have rank equals 1 (axes shape: ",
                          axes_shape,
                          ").");

    NODE_VALIDATION_CHECK(
        this,
        static_cast<size_t>(axes_shape[0]) >= 1 &&
            static_cast<size_t>(axes_shape[0]) <= static_cast<size_t>(input_shape.rank()),
        "Number of elements of axes must be >= 1 and <= argument rank (axes_shape[0]: ",
        axes_shape[0],
        ").");
}

shared_ptr<Node> op::LRN::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::LRN>(new_args.at(0), new_args.at(1), m_alpha, m_beta, m_bias, m_size);
}

void op::LRN::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    throw ngraph_error("NYI");
}
