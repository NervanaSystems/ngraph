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
#include <numeric>

#include "ngraph/op/fused/log_softmax.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::LogSoftmax::type_info;

op::LogSoftmax::LogSoftmax(const Output<Node>& data, int64_t axis)
    : FusedOp({data})
    , m_axis(axis)
{
    constructor_validate_and_infer_types();
}

NodeVector op::LogSoftmax::decompose_op() const
{
    const auto data = input_value(0);
    const auto data_shape = data.get_shape();

    auto axis = ngraph::normalize_axis(this, m_axis, data_shape.size());

    std::vector<size_t> axes(data_shape.size() - axis);
    std::iota(std::begin(axes), std::end(axes), axis);

    auto softmax = std::make_shared<ngraph::op::Softmax>(data, axes);

    return {std::make_shared<ngraph::op::Log>(softmax)};
}

shared_ptr<Node> op::LogSoftmax::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<LogSoftmax>(new_args.at(0), m_axis);
}
