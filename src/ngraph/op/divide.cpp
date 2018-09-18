//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

op::Divide::Divide(const shared_ptr<Node>& arg0, const shared_ptr<Node>& arg1)
    : BinaryElementwiseArithmetic("Divide", arg0, arg1)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::Divide::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Divide>(new_args.at(0), new_args.at(1));
}

void op::Divide::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = get_argument(0);
    auto y = get_argument(1);

    adjoints.add_delta(x, delta / y);
    adjoints.add_delta(y, -delta * shared_from_this() / y);
}

void op::Divide::validate_and_infer_types()
{
    validate_and_infer_elementwise_arithmetic();

    // Static value propagation.
    if (get_inputs().at(0).get_output().has_static_value() &&
        get_inputs().at(1).get_output().has_static_value())
    {
        const StaticValue& sv0 = get_inputs().at(0).get_output().get_static_value();
        const StaticValue& sv1 = get_inputs().at(1).get_output().get_static_value();

        // If validation succeeded we should be safe to assume that both args'
        // static values have the same length, but we will check anyway.
        NODE_VALIDATION_ASSERT(this, sv0.size() == sv1.size())
            << "Internal nGraph error: size of input static values does not match "
            << "(static value 0: " << sv0 << ", static value 1: " << sv1 << ").";

        StaticValue result;

        // TODO(amprocte): Would be reasonable to do a div-by-zero check here.
        for (size_t i = 0; i < sv0.size(); i++)
        {
            result.push_back(sv0[i] / sv1[i]);
        }

        get_outputs().at(0).set_static_value(result);
    }
    else
    {
        clear_output_static_value(0);
    }
}

shared_ptr<Node> ngraph::operator/(const shared_ptr<Node> arg0, const shared_ptr<Node> arg1)
{
    return make_shared<op::Divide>(arg0, arg1);
}
