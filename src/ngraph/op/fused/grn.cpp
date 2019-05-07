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
#include <algorithm>
#include <memory>

#include "grn.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/util/broadcasting.hpp"
#include "ngraph/op/util/norm.hpp"
#include "ngraph/op/util/reshape.hpp"

using namespace std;
using namespace ngraph;

op::GRN::GRN(const shared_ptr<Node>& data, float bias)
    : FusedOp("GRN", {data})
    , m_bias(bias)
{
    const auto& data_shape = data->get_shape();

    NODE_VALIDATION_CHECK(this,
                          (data_shape.size() >= 2 && data_shape.size() <= 4),
                          "Input tensor rank must be 2, 3 or 4 dimensional (actual input shape: ",
                          data_shape,
                          ").");

    constructor_validate_and_infer_types();
}

NodeVector op::GRN::decompose_op() const
{
    const auto input_node{get_argument(0)};
    const auto& input_shape{input_node->get_shape()};

    const auto data{input_node};
    // reshape to 4D tensor
    if (input_shape.size() != 4)
    {
        Shape data_shape{input_shape};
        data_shape.resize(4);
        fill(begin(data_shape), next(begin(data_shape), input_shape.size()), size_t{1});
        data = util::reshape(data, data_shape);
    }

    // calculate l2 norm across channels
    shared_ptr<Node> norm = l2_norm(data, AxisSet{2, 3}, m_bias);
    norm = make_broadcast_node(norm, data->get_shape(), 0);
    data = data / norm;

    // get back original input tensor rank
    if (input_shape.size() != 4)
    {
        data = util::reshape(data, input_shape);
    }

    return data;
}

shared_ptr<Node> op::GRN::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<GRN>(new_args.at(0), m_bias);
}
