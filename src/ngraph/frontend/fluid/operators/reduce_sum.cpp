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

#include <cmath>

#include "ngraph/builder/make_constant.hpp"
#include "ngraph/frontend/fluid/operators/reduce_sum.hpp"

using namespace std;
using namespace ngraph::fluid;

constexpr NodeTypeInfo ReduceSum::type_info;

ReduceSum::ReduceSum(const Output<Node>& x, const vector<int>& dim, bool reduce_all, bool keep_dim)
    : FusedOp({x})
{
    constructor_validate_and_infer_types();
}

NodeVector ReduceSum::decompose_op() const
{
    return {};
}

shared_ptr<Node> ReduceSum::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<Gelu>(new_args.at(0));
}

void ReduceSum::validate_and_infer_types()
{
    element::Type input_element_type = get_input_element_type(0);

    NODE_VALIDATION_CHECK(this,
                          input_element_type.is_dynamic() || input_element_type.is_real(),
                          "Argument element type must be f16, bf16, f32, f64 or dynamic (got ",
                          input_element_type,
                          ").");
}

constexpr NodeTypeInfo ReduceSumGrad::type_info;

ReduceSumGrad::ReduceSumGrad(const Output<Node>& x, const vector<int>& dim, bool reduce_all, bool keep_dim)
    : FusedOp({x})
{
    constructor_validate_and_infer_types();
}

void ReduceSumGrad::validate_and_infer_types()
{
    element::Type input_element_type = get_input_element_type(0);

    NODE_VALIDATION_CHECK(this,
                          input_element_type.is_dynamic() || input_element_type.is_real(),
                          "Argument element type must be f16, bf16, f32, f64 or dynamic (got ",
                          input_element_type,
                          ").");
}

shared_ptr<Node> ReduceSumGrad::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<GeluBackpropFactor>(new_args.at(0));
}

NodeVector ReduceSumGrad::decompose_op() const
{
    auto one = builder::make_constant(x.get_element_type(), x.get_shape(), 1.0);
    auto pi = 4.0 * std::atan(1);
    auto inv_sqrt_two_pi =
        builder::make_constant(x.get_element_type(), x.get_shape(), 1.0 / std::sqrt(2.0 * pi));
    auto sqrt_half = builder::make_constant(x.get_element_type(), x.get_shape(), std::sqrt(0.5));

    auto e1 = half * (one + make_shared<op::Erf>(x * sqrt_half));
    auto e2 = x * make_shared<op::Exp>(x * x * (-half)) * inv_sqrt_two_pi;
    return {e1 + e2};
}
