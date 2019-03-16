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

#include "ngraph/op/tan.hpp"
#include "ngraph/op/cos.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/multiply.hpp"

using namespace std;
using namespace ngraph;

op::Tan::Tan(const NodeOutput& arg)
    : UnaryElementwiseArithmetic("Tan", arg)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::Tan::copy_with_new_source_outputs(const OutputVector& new_source_outputs) const
{
    check_new_source_outputs_count(this, new_source_outputs);
    return make_shared<Tan>(new_source_outputs.at(0));
}

void op::Tan::build_backprop(autodiff::Adjoints& adjoints, const OutputVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = get_input_source_output(0);

    auto c = make_shared<op::Cos>(x);

    adjoints.add_output_delta(x, delta / (c * c));
}
