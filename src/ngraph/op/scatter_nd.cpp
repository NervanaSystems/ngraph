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

#include "ngraph/node.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/scatter_nd.hpp"

using namespace std;
using namespace ngraph;

static int INPUTS = 0;
static int INDICES = 1;
static int UPDATES = 2;

constexpr NodeTypeInfo op::v0::ScatterND::type_info;

op::v0::ScatterND::ScatterND(const Output<Node>& data,
                             const Output<Node>& indices,
                             const Output<Node>& updates)
    : op::util::FusedOp({data, indices, updates})
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v0::ScatterND::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<ScatterND>(new_args.at(0), new_args.at(1), new_args.at(2));
}

void op::v0::ScatterND::pre_validate_and_infer_types()
{
    const static int DATA = 0;
    const static int INDICES = 1;
    const static int UPDATES = 2;

    element::Type data_et = input_value(DATA).get_element_type();
    element::Type indices_et = input_value(INDICES).get_element_type();
    element::Type updates_et = input_value(UPDATES).get_element_type();

    NODE_VALIDATION_CHECK(this,
                          indices_et == element::i32 || indices_et == element::i64,
                          "Indices element type must be i64 or i32.");

    NODE_VALIDATION_CHECK(this,
                          data_et == updates_et,
                          "Updates element type must be the same as element type of data.");

    const PartialShape& data_ps = get_input_partial_shape(DATA);
    const PartialShape& indices_ps = get_input_partial_shape(INDICES);
    const PartialShape& updates_ps = get_input_partial_shape(UPDATES);

    if (data_ps.rank().is_static())
    {
        const size_t data_rank = data_ps.rank().get_length();
        NODE_VALIDATION_CHECK(this, data_rank >= 1, "Data rank is expected to be at least 1.");
    }

    if (indices_ps.rank().is_static())
    {
        const size_t indices_rank = indices_ps.rank().get_length();

        NODE_VALIDATION_CHECK(
            this, indices_rank >= 1, "Indices rank is expected to be at least 1.");
    }

    if (indices_ps.rank().is_static() && data_ps.rank().is_static())
    {
        const size_t indices_rank = indices_ps.rank().get_length();
        const size_t last_dim_pos = indices_rank - 1;
        const Dimension indices_last_dim = indices_ps[last_dim_pos];
        if (indices_last_dim.is_static())
        {
            const size_t indices_last_dim_value = indices_last_dim.get_length();
            const size_t data_rank = data_ps.rank().get_length();
            NODE_VALIDATION_CHECK(this,
                                  indices_last_dim_value <= data_rank,
                                  "Last dimension of indices can be at most the rank of data.");

            if (updates_ps.rank().is_static())
            {
                const size_t expected_updates_rank =
                    data_rank + indices_rank - indices_last_dim_value - 1;

                NODE_VALIDATION_CHECK(
                    this,
                    updates_ps.rank().get_length() == expected_updates_rank,
                    "Updates rank is expected to be equal data_rank + indices_rank - "
                    "indices_shape[-1] - 1.");
            }
        }
    }

    set_output_type(0, data_et, data_ps);
}

NodeVector op::v0::ScatterND::decompose_op() const
{
    const auto data = input_value(0);
    const auto indices = input_value(1);
    const auto updates = input_value(2);

    const Shape& data_shape = data.get_shape();
    const Shape& updates_shape = updates.get_shape();

    element::Type data_et = data.get_element_type();

    // Create a boolean mask that matches the data tensor shape and
    // contains 'true' values in the positions indicated by 'indices'
    // and 'false' values everywhere else.

    const auto true_values = op::Constant::create(element::i64, updates_shape, {1});
    const auto false_values = op::Constant::create(element::i64, data_shape, {0});

    const auto mask = std::make_shared<op::v0::ScatterND>(false_values, indices, true_values);

    const auto mask_bool = std::make_shared<op::v0::Convert>(mask, element::boolean);

    const auto zeros = op::Constant::create(data_et, data_shape, {0});

    // Create an intermediate node that will contain the original data and
    // zeros in the positions indicated by indices.

    const auto intermediate = std::make_shared<op::v0::Select>(mask_bool, zeros, data);

    return {std::make_shared<op::v0::ScatterND>(intermediate, indices, updates)};
}

constexpr NodeTypeInfo op::v3::ScatterND::type_info;

shared_ptr<Node> op::v3::ScatterND::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<ScatterND>(
        new_args.at(INPUTS), new_args.at(INDICES), new_args.at(UPDATES));
}

void op::v3::ScatterND::validate_and_infer_types()
{
    element::Type inputs_et = get_input_element_type(INPUTS);
    element::Type indices_et = get_input_element_type(INDICES);
    element::Type updates_et = get_input_element_type(UPDATES);

    const PartialShape& inputs_shape = get_input_partial_shape(INPUTS);
    const PartialShape& indices_shape = get_input_partial_shape(INDICES);
    const PartialShape& updates_shape = get_input_partial_shape(UPDATES);

    NODE_VALIDATION_CHECK(this,
                          indices_et == element::i32 || indices_et == element::i64,
                          "Indices element type must be i64 or i32");

    NODE_VALIDATION_CHECK(
        this, updates_et == inputs_et, "Updates element type must be the same as inputs");

    NODE_VALIDATION_CHECK(this,
                          indices_shape.rank().is_dynamic() ||
                              indices_shape.rank().get_length() >= 1,
                          "Indices rank is expected to be at least 1");

    NODE_VALIDATION_CHECK(this,
                          inputs_shape.rank().is_dynamic() || indices_shape.rank().is_dynamic() ||
                              indices_shape[indices_shape.rank().get_length() - 1].get_length() <=
                                  inputs_shape.rank().get_length(),
                          "Last dimension of indices can be at most the rank of inputs");

    NODE_VALIDATION_CHECK(
        this,
        inputs_shape.rank().is_dynamic() || indices_shape.rank().is_dynamic() ||
            updates_shape.rank().is_dynamic() ||
            updates_shape.rank().get_length() ==
                indices_shape.rank().get_length() + inputs_shape.rank().get_length() -
                    indices_shape[indices_shape.rank().get_length() - 1].get_length() - 1,
        "Rank of updates must be rank of inputs + rank of indices - last dimension of indices - 1");

    bool compatible = true;
    if (inputs_shape.is_static() && indices_shape.is_static() && updates_shape.is_static())
    {
        size_t indices_rank = indices_shape.rank().get_length();
        size_t updates_rank = updates_shape.rank().get_length();
        for (size_t i = 0; i < indices_rank - 1; i++)
        {
            compatible = compatible && updates_shape[i].same_scheme(indices_shape[i]);
        }
        size_t j = indices_shape[indices_rank - 1].get_length();
        for (size_t i = indices_rank - 1; i < updates_rank; i++, j++)
        {
            compatible = compatible && updates_shape[i].same_scheme(inputs_shape[j]);
        }
    }

    NODE_VALIDATION_CHECK(
        this,
        compatible,
        "Updates shape must be indices_shape[:-1] + inputs_shape[indices.shape[-1]:]");

    set_output_type(0, inputs_et, inputs_shape);
}
