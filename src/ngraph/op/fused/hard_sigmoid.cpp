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

const string op::HardSigmoid::type_name{"HardSigmoid"};

op::HardSigmoid::HardSigmoid(const Output<Node>& data, float alpha, float beta)
    : FusedOp({data})
    , m_alpha(alpha)
    , m_beta(beta)
{
    constructor_validate_and_infer_types();
}

NodeVector op::HardSigmoid::decompose_op() const
{
    auto data = input(0).get_source_output();
    auto data_shape = data.get_shape();
    size_t elem_count = shape_size(data_shape);

    std::shared_ptr<ngraph::Node> alpha_node = ngraph::op::Constant::create<float>(
        data.get_element_type(), data_shape, std::vector<float>(elem_count, m_alpha));

    std::shared_ptr<ngraph::Node> beta_node = ngraph::op::Constant::create<float>(
        data.get_element_type(), data_shape, std::vector<float>(elem_count, m_beta));

    std::shared_ptr<ngraph::Node> one_node = ngraph::op::Constant::create<float>(
        data.get_element_type(), data_shape, std::vector<float>(elem_count, 1.0));

    std::shared_ptr<ngraph::Node> zero_node = ngraph::op::Constant::create<float>(
        data.get_element_type(), data_shape, std::vector<float>(elem_count, 0.0));

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
