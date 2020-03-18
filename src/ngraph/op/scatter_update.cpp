//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "ngraph/op/scatter_update.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v2::ScatterUpdate::type_info;

shared_ptr<Node> op::v2::ScatterUpdate::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<ScatterUpdate>(new_args.at(0), new_args.at(1), new_args.at(2), get_axis());
}

op::v2::ScatterUpdate::ScatterUpdate(const Output<Node>& inputs,
                                     const Output<Node>& indices,
                                     const Output<Node>& updates,
                                     const int32_t axis)
    : util::Scatter(inputs, indices, updates, axis)
{
    constructor_validate_and_infer_types();
}
