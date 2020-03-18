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

constexpr NodeTypeInfo op::v3::ScatterElementsUpdate::type_info;

op::v3::ScatterElementsUpdate::ScatterElementsUpdate(const Output<Node>& data,
                                                     const Output<Node>& indices,
                                                     const Output<Node>& updates,
                                                     const Output<Node>& axis)
    : Op({data, indices, updates, axis})
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v3::ScatterElementsUpdate::visit_attributes(AttributeVisitor& visitor)
{
    return true;
}

void op::v3::ScatterElementsUpdate::validate_and_infer_types()
{
    element::Type data_et = get_input_element_type(0);
    element::Type indices_et = get_input_element_type(1);
    element::Type updates_et = get_input_element_type(2);
    element::Type axis_et = get_input_element_type(3);

    const PartialShape& data_shape = get_input_partial_shape(0);
    const PartialShape& indices_shape = get_input_partial_shape(1);
    const PartialShape& updates_shape = get_input_partial_shape(2);
    const PartialShape& axis_shape = get_input_partial_shape(3);

    NODE_VALIDATION_CHECK(this,
                          indices_et == element::i32 || indices_et == element::i64,
                          "Indices element type must be i64 or i32");

    NODE_VALIDATION_CHECK(this,
                          axis_et == element::i32 || axis_et == element::i64,
                          "Axis element type must be i64 or i32");

    NODE_VALIDATION_CHECK(this,
                          axis_shape.compatible(PartialShape{}) ||
                              axis_shape.compatible(PartialShape{1}),
                          "Axis input shape are required to be scalar or 1D tensor ",
                          "Got: ",
                          axis_shape,
                          " and: ",
                          axis_shape);

    NODE_VALIDATION_CHECK(this,
                          indices_shape.compatible(updates_shape),
                          "Indices and updates input shapes are required to be the same ",
                          "Got: ",
                          indices_shape,
                          " and: ",
                          updates_shape);

    set_output_size(1);
    set_output_type(0, data_et, data_shape);
}

shared_ptr<Node> op::v3::ScatterElementsUpdate::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v3::ScatterElementsUpdate>(
        new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
}
