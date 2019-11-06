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

#include "ngraph/builder/make_constant.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/fused/softplus.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/shape.hpp"

using namespace ngraph;

constexpr NodeTypeInfo op::Softplus::type_info;

op::Softplus::Softplus(const Output<Node>& data)
    : FusedOp({data})
{
    constructor_validate_and_infer_types();
}

NodeVector op::Softplus::decompose_op() const
{
    const auto data = input_value(0);

    const auto zero_node = builder::make_constant(data.get_element_type(), data.get_shape(), 0.f);
    const auto one_node = builder::make_constant(data.get_element_type(), data.get_shape(), 1.f);

    const auto positive_val_node =
        data + std::make_shared<op::Log>(
                   std::make_shared<op::Exp>(std::make_shared<op::Negative>(data)) + one_node);

    const auto negative_val_node =
        std::make_shared<op::Log>(std::make_shared<op::Exp>(data) + one_node);

    const auto condition_node = std::make_shared<op::Greater>(data, zero_node);

    // This equation represents:
    //     x + log(exp(-x) + 1) - for x > 0; to manage exponent overflow,
    //     log(exp(x) + 1)      - elsewhere.
    //
    return {std::make_shared<op::Select>(condition_node, positive_val_node, negative_val_node)};
}

std::shared_ptr<Node> op::Softplus::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return std::make_shared<Softplus>(new_args.at(0));
}
