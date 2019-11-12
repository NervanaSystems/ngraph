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

#include "ngraph/op/cum_sum.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v0::CumSum::type_info;

op::v0::CumSum::CumSum(const Output<Node>& arg,
                       const Output<Node>& axis,
                       int exclusive,
                       int reverse)
    : Op({arg, axis})
    , m_exclusive(exclusive)
    , m_reverse(reverse)
    , m_axis(1)
{
    constructor_validate_and_infer_types();
}

op::v0::CumSum::CumSum(const Output<Node>& arg, const int64_t axis, int exclusive, int reverse)
    : CumSum(arg, op::Constant::create(element::i64, Shape{}, {axis}), exclusive, reverse)
{
}

shared_ptr<Node> op::CumSum::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::v0::CumSum>(new_args.at(0), new_args.at(1), m_exclusive, m_reverse);
}

void op::v0::CumSum::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = input_value(0);
    auto& x_shape = x.get_shape();

    adjoints.add_delta(
        x, make_shared<op::Broadcast>(delta, x_shape, AxisSet{static_cast<size_t>(get_axis())}));
}

shared_ptr<Node> op::v0::CumSum::get_default_value() const
{
    return ngraph::make_constant_from_string("0", get_element_type(), get_shape());
}
