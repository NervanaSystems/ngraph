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

#include "ngraph/op/sum.hpp"
#include "ngraph/op/broadcast.hpp"

using namespace std;
using namespace ngraph;

op::Sum::Sum(const NodeOutput& arg, const AxisSet& reduction_axes)
    : ArithmeticReduction("Sum", arg, reduction_axes)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::Sum::copy_with_new_source_outputs(const OutputVector& new_source_outputs) const
{
    check_new_source_outputs_count(this, new_source_outputs);
    return make_shared<Sum>(new_source_outputs.at(0), m_reduction_axes);
}

void op::Sum::build_backprop(autodiff::Adjoints& adjoints, const OutputVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = get_input_source_output(0);
    auto& x_shape = get_input_shape(0);

    adjoints.add_output_delta(x, make_shared<op::Broadcast>(delta, x_shape, m_reduction_axes));
}
