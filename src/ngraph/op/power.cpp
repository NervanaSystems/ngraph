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

#include "ngraph/op/power.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/multiply.hpp"

using namespace std;
using namespace ngraph;

op::Power::Power(const NodeOutput& arg0, const NodeOutput& arg1)
    : BinaryElementwiseArithmetic("Power", arg0, arg1)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node>
    op::Power::copy_with_new_source_outputs(const OutputVector& new_source_outputs) const
{
    check_new_source_outputs_count(this, new_source_outputs);
    return make_shared<Power>(new_source_outputs.at(0), new_source_outputs.at(1));
}

void op::Power::generate_adjoints(autodiff::Adjoints& adjoints, const OutputVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = get_input_source_output(0);
    auto y = get_input_source_output(1);

    auto log_x = make_shared<op::Log>(x);

    adjoints.add_output_delta(x, delta * y * shared_from_this() / x);
    adjoints.add_output_delta(y, delta * shared_from_this() * log_x);
}
