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

#include "ngraph/op/add.hpp"

using namespace std;
using namespace ngraph;

// ------------------------------- v0 ------------------------------------------

constexpr NodeTypeInfo op::v0::Add::type_info;

op::v0::Add::Add(const Output<Node>& arg0,
                 const Output<Node>& arg1,
                 const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v0::Add::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::v0::Add>(new_args.at(0), new_args.at(1), this->get_autob());
}

bool op::v0::Add::visit_attributes(AttributeVisitor& visitor)
{
    BinaryElementwiseArithmetic::visit_attributes(visitor);
    return true;
}

void op::v0::Add::generate_adjoints(autodiff::Adjoints& adjoints, const OutputVector& deltas)
{
    if (get_autob().m_type != op::AutoBroadcastType::NONE)
    {
        throw ngraph_error("Autodiff not supported with auto broadcasting");
    }

    auto delta = deltas.at(0);

    auto x = input_value(0);
    auto y = input_value(1);

    adjoints.add_delta(x, delta);
    adjoints.add_delta(y, delta);
}

shared_ptr<Node> ngraph::operator+(const Output<Node>& arg0, const Output<Node>& arg1)
{
    return make_shared<op::Add>(arg0, arg1);
}

// ------------------------------- v1 ------------------------------------------

constexpr NodeTypeInfo op::v1::Add::type_info;

op::v1::Add::Add(const Output<Node>& arg0,
                 const Output<Node>& arg1,
                 const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast)
{
    constructor_validate_and_infer_types();
}

bool op::v1::Add::visit_attributes(AttributeVisitor& visitor)
{
    BinaryElementwiseArithmetic::visit_attributes(visitor);
    return true;
}

shared_ptr<Node> op::v1::Add::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::v1::Add>(new_args.at(0), new_args.at(1), this->get_autob());
}

void op::v1::Add::generate_adjoints(autodiff::Adjoints& adjoints, const OutputVector& deltas)
{
    if (get_autob().m_type != op::AutoBroadcastType::NONE)
    {
        throw ngraph_error("Autodiff not supported with auto broadcasting");
    }

    auto delta = deltas.at(0);

    auto x = input_value(0);
    auto y = input_value(1);

    adjoints.add_delta(x, delta);
    adjoints.add_delta(y, delta);
}
