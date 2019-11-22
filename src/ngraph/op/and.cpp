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

#include "ngraph/op/and.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v1::LogicalAnd::type_info;

op::v1::LogicalAnd::LogicalAnd(const Output<Node>& arg0,
                               const Output<Node>& arg1,
                               const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseLogical(arg0, arg1, auto_broadcast)
{
    constructor_validate_and_infer_types();
}

bool op::v1::LogicalAnd::visit_attributes(AttributeVisitor& visitor)
{
    BinaryElementwiseLogical::visit_attributes(visitor);
    return true;
}

shared_ptr<Node> op::v1::LogicalAnd::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v1::LogicalAnd>(new_args.at(0), new_args.at(1), this->get_autob());
}

constexpr NodeTypeInfo op::v0::And::type_info;

op::v0::And::And(const Output<Node>& arg0,
                 const Output<Node>& arg1,
                 const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseLogical(arg0, arg1, auto_broadcast)
{
    constructor_validate_and_infer_types();
}

bool op::v0::And::visit_attributes(AttributeVisitor& visitor)
{
    BinaryElementwiseLogical::visit_attributes(visitor);
    return true;
}

shared_ptr<Node> op::v0::And::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v0::And>(new_args.at(0), new_args.at(1), this->get_autob());
}
