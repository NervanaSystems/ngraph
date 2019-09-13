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

#include "ngraph/op/divide.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Divide::type_info;

op::Divide::Divide(const Output<Node>& arg0,
                   const Output<Node>& arg1,
                   const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast)
{
    constructor_validate_and_infer_types();
}

op::Divide::Divide(const Output<Node>& arg0,
                   const Output<Node>& arg1,
                   bool pythondiv,
                   const AutoBroadcastSpec& auto_broadcast)
    : BinaryElementwiseArithmetic(arg0, arg1, auto_broadcast)
    , m_pythondiv(pythondiv)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::Divide::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Divide>(
        new_args.at(0), new_args.at(1), this->is_pythondiv(), this->get_autob());
}

void op::Divide::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    if (get_autob().m_type != op::AutoBroadcastType::NONE)
    {
        throw ngraph_error("Autodiff not supported with auto broadcasting");
    }

    auto delta = deltas.at(0);

    auto x = input_value(0);
    auto y = input_value(1);

    adjoints.add_delta(x, delta / y);
    adjoints.add_delta(y, -delta * shared_from_this() / y);
}

shared_ptr<Node> ngraph::operator/(const Output<Node>& arg0, const Output<Node>& arg1)
{
    return make_shared<op::Divide>(arg0, arg1);
}
