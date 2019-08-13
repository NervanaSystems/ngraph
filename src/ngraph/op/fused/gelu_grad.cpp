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

#include "ngraph/op/fused/gelu_grad.hpp"
#include "ngraph/builder/make_constant.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/erf.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/subtract.hpp"

using namespace std;
using namespace ngraph;

const string op::GeluGrad::type_name{"GeluGrad"};

op::GeluGrad::GeluGrad(const Output<Node>& x)
    : FusedOp({x})
{
    constructor_validate_and_infer_types();
}

void op::GeluGrad::pre_validate_and_infer_types()
{
    element::Type input_element_type = get_input_element_type(0);

    NODE_VALIDATION_CHECK(this,
                          input_element_type.is_dynamic() || input_element_type == element::f32 ||
                              input_element_type == element::f64 ||
                              input_element_type == element::f16 ||
                              input_element_type == element::bf16,
                          "Argument element type must be f16, bf16, f32, f64 or dynamic (got ",
                          input_element_type,
                          ").");
}

shared_ptr<Node> op::GeluGrad::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<GeluGrad>(new_args.at(0));
}

NodeVector op::GeluGrad::decompose_op() const
{
    auto x = get_argument(0);

    // 0.5 * (1 + erf( x * sqrt(1/2))
    // + [x * exp (-x^2/2)] / sqrt(2 * pi)
    auto half = builder::make_constant(x->get_element_type(), x->get_shape(), 0.5);
    auto one = builder::make_constant(x->get_element_type(), x->get_shape(), 1.0);
    auto pi = 4.0 * std::atan(1);
    auto inv_sqrt_two_pi =
        builder::make_constant(x->get_element_type(), x->get_shape(), 1.0 / std::sqrt(2.0 * pi));
    auto sqrt_half = builder::make_constant(x->get_element_type(), x->get_shape(), std::sqrt(0.5));

    auto e1 = half * (one + make_shared<op::Erf>(x * sqrt_half));
    auto e2 = x * make_shared<op::Exp>(x * x * (-half)) * inv_sqrt_two_pi;
    return {e1 + e2};
}
