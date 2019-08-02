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
#include "scale_shift.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/util/broadcasting.hpp"

using namespace std;
using namespace ngraph;

op::ScaleShift::ScaleShift(const std::shared_ptr<ngraph::Node>& data,
                           const std::shared_ptr<ngraph::Node>& scale,
                           const std::shared_ptr<ngraph::Node>& shift)
    : FusedOp("ScaleShift", {data, scale, shift})
{
    constructor_validate_and_infer_types();
}

NodeVector op::ScaleShift::decompose_op() const
{
    auto data = input(0).get_source_output();
    auto scale = input(1).get_source_output();
    auto shift = input(2).get_source_output();

    // broadcast all data
    auto broadcasted_nodes = numpy_style_broadcast_values({data, scale, shift});
    data = broadcasted_nodes[0];
    scale = broadcasted_nodes[1];
    shift = broadcasted_nodes[2];

    return {scale * data + shift};
}

shared_ptr<Node> op::ScaleShift::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 3)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<ScaleShift>(new_args.at(0), new_args.at(1), new_args.at(2));
}
