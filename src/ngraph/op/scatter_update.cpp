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

constexpr NodeTypeInfo op::v3::ScatterUpdate::type_info;

op::v3::ScatterUpdate::ScatterUpdate(const Output<Node>& inputs,
                                     const Output<Node>& indices,
                                     const Output<Node>& updates,
                                     const Output<Node>& axis)
    : util::Scatter(inputs, indices, updates, axis)
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v3::ScatterUpdate::visit_attributes(AttributeVisitor& visitor)
{
    return true;
}

// TODO: copyt_with_new_inputs
