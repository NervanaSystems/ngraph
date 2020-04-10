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
#include "ngraph/op/constant.hpp"
#include "ngraph/validation_util.hpp"

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

bool op::v3::ScatterElementsUpdate::visit_attributes(AttributeVisitor& visitor)
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
                          indices_et.is_integral(),
                          "Indices element type must be integral_number, but is: ",
                          indices_et);

    NODE_VALIDATION_CHECK(this,
                          axis_et.is_integral(),
                          "Axis element type must be integral_number, but is: ",
                          axis_et);

    NODE_VALIDATION_CHECK(this,
                          data_et == updates_et,
                          "Data type and updates type are required to be the same. ",
                          "Got: ",
                          data_et,
                          " and: ",
                          updates_et);

    NODE_VALIDATION_CHECK(this,
                          axis_shape.compatible(PartialShape{}) ||
                              axis_shape.compatible(PartialShape{1}),
                          "Axis input shape are required to be scalar or 1D tensor. ",
                          "Got: ",
                          axis_shape);

    NODE_VALIDATION_CHECK(this,
                          indices_shape.rank().compatible(data_shape.rank()),
                          "Indices rank and data rank are required to be equal. ",
                          "Got: ",
                          indices_shape.rank(),
                          " and: ",
                          data_shape.rank());

    NODE_VALIDATION_CHECK(this,
                          indices_shape.compatible(updates_shape),
                          "Indices and updates input shapes are required to be equal. ",
                          "Got: ",
                          indices_shape,
                          " and: ",
                          updates_shape);

    if (input_value(3).get_node_shared_ptr()->is_constant() && data_shape.rank().is_static())
    {
        const auto axis_input = as_type_ptr<op::v0::Constant>(input_value(3).get_node_shared_ptr());
        auto axis = axis_input->cast_vector<int64_t>().at(0);

        int64_t data_rank_length = data_shape.rank().get_length();
        NODE_VALIDATION_CHECK(
            this,
            (-data_rank_length <= axis) && (axis <= data_rank_length - 1),
            "Axis value has to be in range [-r, r-1] where r is rank of data shape. ",
            " Data rank: ",
            data_rank_length,
            ", range:[",
            -data_rank_length,
            ", ",
            data_rank_length - 1,
            "]. Got axis value: ",
            axis);
    }

    if (data_shape.is_dynamic())
    {
        set_input_is_relevant_to_shape(0);
    }

    set_output_size(1);
    set_output_type(0, data_et, data_shape);
}

shared_ptr<Node>
    op::v3::ScatterElementsUpdate::clone_with_new_inputs(const OutputVector& inputs) const
{
    NODE_VALIDATION_CHECK(this,
                          inputs.size() == get_input_size(),
                          "clone_with_new_inputs() required inputs size: ",
                          get_input_size(),
                          "Got: ",
                          inputs.size());

    return make_shared<v3::ScatterElementsUpdate>(
        inputs.at(0), inputs.at(1), inputs.at(2), inputs.at(3));
}
