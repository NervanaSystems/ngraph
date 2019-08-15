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

#include "ngraph/op/fused/squared_difference.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/util/broadcasting.hpp"
#include "ngraph/op/util/fused_op.hpp"

using namespace std;
using namespace ngraph;

const string op::SquaredDifference::type_name{"SquaredDifference"};

op::SquaredDifference::SquaredDifference(const Output<Node>& x1, const Output<Node>& x2)
    : FusedOp({x1, x2})
{
    constructor_validate_and_infer_types();
}

NodeVector op::SquaredDifference::decompose_op() const
{
    const auto x1 = input_value(0);
    const auto x2 = input_value(1);

    const auto broadcasted = numpy_style_broadcast_values({x1, x2});

    const auto difference = broadcasted.at(0) - broadcasted.at(1);

    return {difference * difference};
}

shared_ptr<Node> op::SquaredDifference::copy_with_new_args(const NodeVector& new_args) const
{
    NODE_VALIDATION_CHECK(this,
                          new_args.size() == 2,
                          "Expected 2 elements in new_args for the SquaredDifference op but got ",
                          new_args.size());

    return make_shared<SquaredDifference>(new_args.at(0), new_args.at(1));
}
