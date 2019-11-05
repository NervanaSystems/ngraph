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

#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/fused/hard_sigmoid.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::HardSigmoid::type_info;

op::HardSigmoid::HardSigmoid(const Output<Node>& data, float alpha, float beta)
    : FusedOp({data})
    , m_alpha(alpha)
    , m_beta(beta)
{
    constructor_validate_and_infer_types();
}

op::HardSigmoid::HardSigmoid(const Output<Node>& data,
                             const Output<Node>& alpha,
                             const Output<Node>& beta)
    : FusedOp({data, alpha, beta})
{
    constructor_validate_and_infer_types();
}

void op::HardSigmoid::pre_validate_and_infer_types()
{
    if (get_input_size() > 1)
    {
        const auto& alpha_pshape = get_input_partial_shape(1);
        const auto& beta_pshape = get_input_partial_shape(2);
        NODE_VALIDATION_CHECK(this,
                              alpha_pshape.is_static() && beta_pshape.is_static(),
                              "Both alpha and beta inputs must have static shapes.");

        const Shape alpha_shape = alpha_pshape.to_shape();

        NODE_VALIDATION_CHECK(this,
                              is_vector(alpha_shape) && alpha_shape[0] == 1,
                              "A vector with a single element expected for the alpha input. Got: ",
                              alpha_shape);

        const Shape beta_shape = beta_pshape.to_shape();

        NODE_VALIDATION_CHECK(this,
                              is_vector(beta_shape) && beta_shape[0] == 1,
                              "A vector with a single element expected for the beta input. Got: ",
                              beta_shape);

        const auto& data_et = input(0).get_element_type();
        const auto& alpha_et = input(1).get_element_type();
        const auto& beta_et = input(2).get_element_type();

        NODE_VALIDATION_CHECK(
            this,
            data_et == alpha_et && data_et == beta_et,
            "The element types of both alpha and beta inputs must match the data input type.");
    }
}

NodeVector op::HardSigmoid::decompose_op() const
{
    const auto data = input_value(0);
    const auto data_shape = data.get_shape();
    const size_t elem_count = shape_size(data_shape);

    std::shared_ptr<ngraph::Node> one_node = ngraph::op::Constant::create<float>(
        data.get_element_type(), data_shape, std::vector<float>(elem_count, 1.0f));

    std::shared_ptr<ngraph::Node> zero_node = ngraph::op::Constant::create<float>(
        data.get_element_type(), data_shape, std::vector<float>(elem_count, 0.0f));

    std::shared_ptr<ngraph::Node> alpha_node, beta_node;

    if (get_input_size() > 1)
    {
        alpha_node = input_value(1).get_node_shared_ptr();
        beta_node = input_value(2).get_node_shared_ptr();
    }
    else
    {
        alpha_node = ngraph::op::Constant::create<float>(
            data.get_element_type(), data_shape, std::vector<float>(elem_count, m_alpha));

        beta_node = ngraph::op::Constant::create<float>(
            data.get_element_type(), data_shape, std::vector<float>(elem_count, m_beta));
    }

    return {std::make_shared<op::Minimum>(
        std::make_shared<op::Maximum>(alpha_node * data + beta_node, zero_node), one_node)};
}

shared_ptr<Node> op::HardSigmoid::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<HardSigmoid>(new_args.at(0), m_alpha, m_beta);
}
