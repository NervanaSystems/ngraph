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

#include "ngraph/op/fused/softsign.hpp"
#include "ngraph/builder/make_constant.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/util/broadcasting.hpp"
#include "ngraph/shape.hpp"

using namespace ngraph;

constexpr NodeTypeInfo op::Softsign::type_info;

op::Softsign::Softsign(const Output<Node>& data)
    : FusedOp({data})
{
    constructor_validate_and_infer_types();
}

NodeVector op::Softsign::decompose_op() const
{
    const auto data = input_value(0);
    const auto data_shape = data.get_shape();
    const auto one_node = op::Constant::create(data.get_element_type(), data_shape, {1});

    return {data / (std::make_shared<op::Abs>(data) + one_node)};
}

std::shared_ptr<Node> op::Softsign::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return std::make_shared<Softsign>(new_args.at(0));
}