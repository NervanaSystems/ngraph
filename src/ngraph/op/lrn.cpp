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

#include "ngraph/op/lrn.hpp"
#include "ngraph/op/multiply.hpp"

using namespace std;
using namespace ngraph;

op::LRN::LRN(const std::shared_ptr<Node>& arg, double alpha, double beta, double bias, size_t nsize)
    : UnaryElementwiseArithmetic("LRN", arg)
    , m_alpha(alpha)
    , m_beta(beta)
    , m_bias(bias)
    , m_size(nsize)
{
    constructor_validate_and_infer_types();
}

void op::LRN::validate_and_infer_types()
{
    UnaryElementwiseArithmetic::validate_and_infer_types();

    const PartialShape& input_shape = get_input_partial_shape(0);

    NODE_VALIDATION_ASSERT(
        this, input_shape.rank().is_dynamic() || static_cast<size_t>(input_shape.rank()) >= 3)
        << "Argument must have rank >= 3 (argument shape: " << input_shape << ").";
}

shared_ptr<Node> op::LRN::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::LRN>(new_args.at(0), m_alpha, m_beta, m_bias, m_size);
}

void op::LRN::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    throw ngraph_error("NYI");
}
