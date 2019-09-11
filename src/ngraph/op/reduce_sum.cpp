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

#include "ngraph/op/reduce_sum.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/broadcast.hpp"

using namespace std;
using namespace ngraph;

const string op::v1::ReduceSum::type_name{ "Sum" };

op::v1::ReduceSum::ReduceSum(const Output<Node>& arg, const AxisSet& reduction_axes, bool keep_dims)
    : ArithmeticReduction(arg, reduction_axes)
    , m_keep_dims{ keep_dims }
{
    constructor_validate_and_infer_types();
}

op::v1::ReduceSum::ReduceSum(const Output<Node>& arg,
    const Output<Node>& reduction_axes,
    bool keep_dims)
    : ArithmeticReduction(arg, reduction_axes)
    , m_keep_dims{ keep_dims }
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v1::ReduceSum::get_default_value() const
{
    return ngraph::make_constant_from_string("0", get_element_type(), get_shape());
}

shared_ptr<Node> op::v1::ReduceSum::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<ReduceSum>(new_args.at(0), new_args.at(1), m_keep_dims);
}

void op::v1::ReduceSum::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = input_value(0);
    auto& x_shape = x.get_shape();

    adjoints.add_delta(x, make_shared<op::Broadcast>(delta, x_shape, get_reduction_axes()));
}
