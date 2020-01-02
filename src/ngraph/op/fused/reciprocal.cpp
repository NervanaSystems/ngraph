//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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
#include "ngraph/op/fused/reciprocal.hpp"

#include "ngraph/op/constant.hpp"
#include "ngraph/op/divide.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Reciprocal::type_info;

op::Reciprocal::Reciprocal(const Output<Node>& data)
    : FusedOp({data})
{
    constructor_validate_and_infer_types();
}

NodeVector op::Reciprocal::decompose_op() const
{
    auto data = input_value(0);
    auto one_node = op::Constant::create(data.get_element_type(), data.get_shape(), {1});
    return {make_shared<op::v1::Divide>(one_node, data)};
}

shared_ptr<Node> op::Reciprocal::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Reciprocal>(new_args.at(0));
}
