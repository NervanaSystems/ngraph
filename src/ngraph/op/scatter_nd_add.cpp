//*****************************************************************************
// Copyright 2019 Intel Corporation
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

#include "ngraph/op/scatter_nd_add.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

static int INPUTS = 0;
static int INDICES = 1;
static int UPDATES = 2;

shared_ptr<Node> op::ScatterNDAdd::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<ScatterAdd>(new_args.at(INPUTS), new_args.at(INDICES), new_args.at(UPDATES));
}

void op::GatherND::validate_and_infer_types()
{
    element::Type inputs_et = get_input_element_type(INPUTS);
    element::Type indices_et = get_input_element_type(INDICES);
    element::Type updates_et = get_input_element_type(UPDATES);

    const PartialShape& inputs_shape = get_input_partial_shape(INPUTS);
    const PartialShape& indices_shape = get_input_partial_shape(INDICES);

    NODE_VALIDATION_CHECK(this,
                          indices_et == element::i32 || indices_et == element::i64,
                          "Indices element type must be i64 or i32");

    NODE_VALIDATION_CHECK(this,
                          updates_et == inputs_et,
                          "Updates element type must be the same as inputs");

    NODE_VALIDATION_CHECK(this,
                          indices_shape.rank().is_dynamic() ||
                              static_cast<size_t>(indices_shape.rank()) >= 1,
                          "indices rank is expected to be at least 1");

    NODE_VALIDATION_CHECK(this,
                          inputs_shape.rank().is_dynamic() ||
                              static_cast<size_t>(inputs_shape.rank()) >= 1,
                          "inputs rank is expected to be at least 1");

    NODE_VALIDATION_CHECK(
        this,
        inputs_shape.rank().is_dynamic() || indices_shape.rank().is_dynamic() ||
            static_cast<size_t>(indices_shape[static_cast<size_t>(indices_shape.rank()) - 1]) <=
                static_cast<size_t>(params_shape.rank()),
        "last dimension of indices can be at most the rank of inputs");

    NODE_VALIDATION_CHECK(
        this,
        inputs_shape.rank().is_dynamic() || indices_shape.rank().is_dynamic() ||
            updates_shape.rank().is_dynamic() ||
            static_cast<size_t>(updates_shape.rank()) ==
                static_cast<size_t>(indices_shape.rank()) + static_cast<size_t>(inputs_shape.rank()) -
                static_cast<size_t>(indices_shape[static_cast<size_t>(indices_shape.rank()) - 1]) - 1,
        "Rank of updates must be rank of inputs + rank of indices - last dimension of indices - 1");

    // TODO: check updates shape

    set_output_type(0, inputs_et, inputs_shape);
}
