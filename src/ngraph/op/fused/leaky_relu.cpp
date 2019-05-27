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

#include "ngraph/op/fused/leaky_relu.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/util/broadcasting.hpp"

using namespace std;
using namespace ngraph;

op::LeakyRelu::LeakyRelu(const shared_ptr<Node>& data, const shared_ptr<Node>& alpha)
    : FusedOp("LeakyRelu", {data, alpha})
{
    constructor_validate_and_infer_types();
}

NodeVector op::LeakyRelu::decompose_op() const
{
    auto data = get_argument(0);
    auto alpha_node = get_argument(1);

    alpha_node = ngraph::op::numpy_style_broadcast(alpha_node, data->get_shape());
    return {std::make_shared<ngraph::op::Maximum>(data * alpha_node, data)};
}

shared_ptr<Node> op::LeakyRelu::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 2)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<LeakyRelu>(new_args.at(0), new_args.at(1));
}
