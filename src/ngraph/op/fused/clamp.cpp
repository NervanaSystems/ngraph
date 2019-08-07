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
#include "ngraph/op/fused/clamp.hpp"

#include "ngraph/builder/make_constant.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"

using namespace std;
using namespace ngraph;

const string op::Clamp::type_name{"Clamp"};

op::Clamp::Clamp(const Output<Node>& data, const double min, const double max)
    : FusedOp({data})
    , m_min{min}
    , m_max{max}
{
    constructor_validate_and_infer_types();
}

void op::Clamp::pre_validate_and_infer_types()
{
    NODE_VALIDATION_CHECK(
        this, m_min < m_max, "The 'min' parameter needs to be less than 'max' for Clamp");
}

NodeVector op::Clamp::decompose_op() const
{
    const auto data = input(0).get_source_output();
    const auto data_shape = data.get_shape();

    const auto clamp_min = builder::make_constant(data.get_element_type(), data_shape, m_min);
    const auto clamp_max = builder::make_constant(data.get_element_type(), data_shape, m_max);

    return {std::make_shared<ngraph::op::Minimum>(
        clamp_max, std::make_shared<ngraph::op::Maximum>(clamp_min, data))};
}

shared_ptr<Node> op::Clamp::copy_with_new_args(const NodeVector& new_args) const
{
    NODE_VALIDATION_CHECK(this,
                          new_args.size() == 1,
                          "Expected 1 element in new_args for the Clamp op but got ",
                          new_args.size());

    return make_shared<Clamp>(new_args.at(0), m_min, m_max);
}
