// ----------------------------------------------------------------------------
// Copyright 2018 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include "ngraph/ops/relu.hpp"
#include "ngraph/ops/multiply.hpp"

using namespace std;
using namespace ngraph;

op::Relu::Relu(shared_ptr<Node> arg0)
    : UnaryElementwiseArithmetic("Relu", {arg0})
{
    set_value_type_checked(arg0->get_element_type(), arg0->get_shape());
}

void op::Relu::generate_adjoints(autodiff::Adjoints& adjoints, const std::shared_ptr<Node>& delta)
{
    adjoints.add_delta(get_input_op(0), delta);
}

shared_ptr<Node> op::Relu::copy_with_new_args(const vector<shared_ptr<Node>>& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<Relu>(new_args.at(0));
}
