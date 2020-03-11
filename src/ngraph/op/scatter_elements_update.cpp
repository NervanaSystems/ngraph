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

#include "ngraph/op/scatter_elements_update.hpp"

using namespace ngraph;
using namespace std;

constexpr NodeTypeInfo op::v0::ScatterElementsUpdate::type_info;

op::v0::ScatterElementsUpdate::ScatterElementsUpdate(const Output<Node>& data,
                                                     const Output<Node>& indices,
                                                     const Output<Node>& updates,
                                                     const Output<Node>& axis)
    : Op({data, indices, updates, axis})
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::ScatterElementsUpdate::visit_attributes(AttributeVisitor& visitor)
{
    return true;
}

void op::v0::ScatterElementsUpdate::validate_and_infer_types()
{
}

shared_ptr<Node> op::v0::ScatterElementsUpdate::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v0::ScatterElementsUpdate>(
        new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
}
