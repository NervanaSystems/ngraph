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

#include "ngraph/op/fused/scatter_nd.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/scatter_nd_add.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

static int DATA = 0;
static int INDICES = 1;
static int UPDATES = 2;

constexpr NodeTypeInfo op::ScatterND::type_info;

shared_ptr<Node> op::ScatterND::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<ScatterND>(new_args.at(DATA), new_args.at(INDICES), new_args.at(UPDATES));
}

void op::ScatterND::validate_and_infer_types()
{
    element::Type data_et = get_input_element_type(DATA);
    element::Type indices_et = get_input_element_type(INDICES);
    element::Type updates_et = get_input_element_type(UPDATES);

    const PartialShape& data_shape = get_input_shape(DATA);
    const PartialShape& indices_shape = get_input_shape(INDICES);
    const PartialShape& updates_shape = get_input_shape(UPDATES);

    const size_t data_rank = static_cast<size_t>(data_shape.rank());
    const size_t indices_rank = static_cast<size_t>(indices_shape.rank());
    const size_t updates_rank = static_cast<size_t>(updates_shape.rank());

    const size_t indices_last_dim =
        static_cast<size_t>(indices_shape[static_cast<size_t>(indices_shape.rank()) - 1]);

    NODE_VALIDATION_CHECK(this,
                          indices_et == element::i32 || indices_et == element::i64,
                          "Indices element type must be i64 or i32.");

    NODE_VALIDATION_CHECK(this,
                          data_et == updates_et,
                          "Updates element type must be the same as element type of data. ");

    NODE_VALIDATION_CHECK(this,
                          indices_shape.rank().is_dynamic() || indices_rank >= 1,
                          "Indices rank is expected to be at least 1.");

    NODE_VALIDATION_CHECK(this,
                          data_shape.rank().is_dynamic() || data_rank >= 1,
                          "Data rank is expected to be at least 1.");

    NODE_VALIDATION_CHECK(this,
                          data_shape.rank().is_dynamic() || indices_shape.rank().is_dynamic() ||
                              indices_last_dim <= data_rank,
                          "Last dimension of indices can be at most the rank of data.");

    const size_t expected_updates_rank = data_rank + indices_rank - indices_last_dim - 1;

    NODE_VALIDATION_CHECK(
        this,
        data_shape.rank().is_dynamic() || updates_rank == expected_updates_rank,
        "update rank is expected to be equal data_rank + indices_rank - indices_shape[-1] - 1");

    set_output_type(0, data_et, data_shape);
}

NodeVector op::ScatterND::decompose_op() const
{
    const auto data = input_value(DATA);
    const auto indices = input_value(INDICES);
    const auto updates = input_value(UPDATES);

    const Shape& data_shape = get_input_shape(DATA);
    const Shape& updates_shape = get_input_shape(UPDATES);

    element::Type data_et = get_input_element_type(DATA);

    const auto false_values = op::Constant::create(element::Type_t::boolean, data_shape, {false});
    const auto true_values = op::Constant::create(element::Type_t::boolean, updates_shape, {true});

    const auto mask = std::make_shared<op::v0::ScatterNDAdd>(true_values, indices, false_values);

    const auto zeros = op::Constant::create(data_et, data_shape, {0});

    const auto intermediate = std::make_shared<op::v0::Select>(mask, data, zeros);

    return {std::make_shared<op::v0::ScatterNDAdd>(intermediate, indices, updates)};
}
